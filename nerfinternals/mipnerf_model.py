# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of mip-NeRF.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

from nerfinternals.nerf_field import aNeRFField


class MipNerfInternalModel(Model):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.scene_contraction = None
        if self.config.spatial_distortion not in "None":
            if self.config.spatial_distortion in "inf":
                self.scene_contraction = SceneContraction(order=float("inf"))
            else:
                self.scene_contraction = SceneContraction()

        self.field = aNeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True,
            spatial_distortion=self.scene_contraction
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        if self.config.background_color in "last_sample":
            self.renderer_rgb = RGBRenderer(background_color="last_sample")
        else:
            self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    @torch.no_grad()
    def forward_activation_layer(self, ray_bundle: RayBundle,
                                 layer: int = 0,
                                 num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_activations_in_layer(ray_bundle, layer=layer, num_samples=num_samples)

    def forward_fine(self, ray_bundle: RayBundle,
                     act_coarse: Optional[torch.Tensor] = None,
                     act_fct: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            weights_coarse: weights for deriving a pdf-density
            act_coarse: activation to derive weights from
            pdf_samples: optional samples to use instead of uniform samples
            act_fct: function to apply to the activations before normalizing
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs_fine(ray_bundle,
                                     act_coarse,
                                     act_fct=act_fct)

    def get_outputs_fine(self, ray_bundle: RayBundle, weights_coarse: torch.Tensor,
                         act_coarse: Optional[torch.Tensor] = None,
                         pdf_samples: Optional[RaySamples] = None,
                         act_fct: Optional[Callable] = None):
        """ derives a density from the activations and uses this to derive a pdf-density

        Args:
            ray_bundle: Ray bundle (parameters)
            weights_coarse: weights for deriving a pdf-density
            act_coarse: activation to derive weights from
            pdf_samples: optional samples to use instead of uniform samples
            act_fct: function to apply to the activations before normalizing

        Returns:
            a dict of images
        """
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        num_samples: int = self.sampler_uniform.num_samples if act_coarse is None else act_coarse.shape[-1]
        ray_samples_uniform = self.sampler_uniform(ray_bundle, num_samples=num_samples)
        # If the activation is not None, use it
        if act_fct is not None:
            weights_ray_act = act_fct(act_coarse)
        else:
            weights_ray_act = torch.relu((act_coarse.mean(-1, keepdim=True) - (act_coarse.std(-1, keepdim=True)/2))
                                         - act_coarse)**2
        # derive weights from the activations
        weights_coarse_act = ray_samples_uniform.get_weights(weights_ray_act.unsqueeze(-1))
        samples: RaySamples = ray_samples_uniform

        # normalize weights to 1
        weights_coarse_act /= weights_coarse_act.sum(-2, keepdim=True)
        # obtain a pdf-distributed samples from the activations
        ray_samples_pdf = self.sampler_pdf(ray_bundle, samples, weights_coarse_act,
                                           num_samples=self.sampler_pdf.num_samples)
        # num_samples=self.sampler_pdf.num_samples + self.sampler_uniform.num_samples // 2)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb": rgb_fine,
            "accumulation": accumulation_fine,
            "depth": depth_fine,
        }
        return outputs

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb_fine = outputs["rgb"]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "ssim": float(fine_ssim.item()),
            "lpips": float(fine_lpips.item()),
        }
        return metrics_dict, None

    def get_activations_in_layer(self,
                                ray_bundle: RayBundle,
                                layer: int,
                                num_samples: Optional[int] = None,):
        """ Records activations for a layer (means over N_h)

        Args:
            ray_bundle: Ray bundle (parameters)
            layer: Layer for which to extract the activaitons

        Returns:
            a dict of images
        """
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        ray_samples_uniform = self.sampler_uniform(ray_bundle, num_samples)

        outs = self.field.get_activation_in_layer(ray_samples_uniform, layer)
        return outs.mean(-1)
