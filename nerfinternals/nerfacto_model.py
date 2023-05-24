"""
Nerfacto implementation, using activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional, Callable
from functools import partial

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from time import time
from nerfinternals.utils.mlp_density_field import MLPDensityField
from nerfinternals.utils.act_proposal import aProposalNetworkSampler
from nerfinternals.utils.ndc_collider import NDCCollider

@dataclass
class NerfactoInternalModelConfig(ModelConfig):
    """Nerfacto Internal Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoInternalModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_ndc_collider: bool = False
    """Whether to disable scene contraction or not."""


class NerfactoInternalModel(Model):
    """Nerfacto Internal model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoInternalModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        # we don't do this here, we (might) use a custom collider
        # super().populate_modules()
        if self.config.enable_collider:
            if self.config.use_ndc_collider:
                assert 'hwf' in self.kwargs
                hwf = self.kwargs.get('hwf')
                self.collider = NDCCollider(
                    h=int(hwf[0]), w=int(hwf[1]), focal=hwf[2]
                )
            else:
                assert self.config.collider_params is not None
                self.collider = NearFarCollider(
                    near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
                )

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = MLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                network = MLPDensityField(
                        self.scene_box.aabb,
                        spatial_distortion=scene_contraction
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = aProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict

    def forward_fine(self,
                     ray_bundle: RayBundle,
                     act_coarse: Optional[torch.Tensor] = None,
                     act_fct: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            act_coarse: pre-computed activation (might be of a lower resolution
            act_fct: function to transform the activations
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs_fine(ray_bundle, act_fct, act_coarse)

    def get_outputs_fine(self,
                         ray_bundle: RayBundle,
                         act_fct: Optional[Callable] = None,
                         act_coarse: Optional[torch.Tensor] = None):
        density_fcts = self.density_fns.copy()
        if act_coarse is not None:
            density_fcts[0] = partial(self.proposal_networks[0].get_density_from_activation,
                                      density_fct=act_fct, activation=act_coarse)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fcts)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_activations_in_layer(self, ray_bundle: RayBundle,
                                 layer: int = 0,
                                 num_samples: Optional[int] = None) -> torch.Tensor:
        act_fns : List[Callable] = []
        for i, prop_net in enumerate(self.proposal_networks):
            if isinstance(prop_net, MLPDensityField) and i == 0:
                act_fns.append(partial(prop_net.get_activations_in_layer, layer=layer))
            else:
                act_fns.append(None)

        activations = self.proposal_sampler.get_activations_in_layer(ray_bundle,
             activations_fns=act_fns, num_samples=num_samples)
        return activations

    @torch.no_grad()
    def forward_activation_layer(self, ray_bundle: RayBundle,
                                 layer: int = 0,
                                 num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_activations_in_layer(ray_bundle, layer=layer, num_samples=num_samples)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_activationinformed(self, camera_ray_bundle: RayBundle,
                                                             layer: Optional[int] = 0,
                                                             upsample: bool = False,
                                                             upsample_res: List[int] = None,
                                                             num_samples: Optional[int] = None,
                                                             act_fct: Optional[Callable] = None) -> (
            Dict[str, torch.Tensor], Dict):
        """Takes in camera parameters and computes the output of the model, with activations.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            layer: analyze activation of this layer
            upsample: whether to use upsampling or not
            upsample_res: resolution of the activation map (if upsampling is enabled)
            num_samples: how many samples for evaluating the activation
            act_fct: what function to use for transformation
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]

        # profiling purposes
        inner_start = time()

        if upsample:
            activation_map: torch.Tensor = torch.zeros(size=upsample_res, device=camera_ray_bundle.origins.device)
            num_samples = upsample_res[-1]
            activation_resolution_factor = [image_width // upsample_res[0], image_height // upsample_res[1]]
        else:
            activation_map: torch.Tensor = torch.zeros(size=[image_width, image_height, num_samples],
                                                       device=camera_ray_bundle.origins.device)
            activation_resolution_factor = [1, 1]

        for i in range(0, activation_map.shape[0] * activation_map.shape[1], num_rays_per_chunk):
            start_idx = i
            end_idx = min(i + num_rays_per_chunk, activation_map.shape[0] * activation_map.shape[1])
            ray_bundle = camera_ray_bundle[::activation_resolution_factor[0],
                         ::activation_resolution_factor[1]].get_row_major_sliced_ray_bundle(start_idx, end_idx)
            activation_map.reshape(-1, num_samples)[start_idx:end_idx] = self.forward_activation_layer(
                ray_bundle=ray_bundle, layer=layer,
                num_samples=num_samples)
        t_act = time() - inner_start
        # another pass, this time with activations as density guide
        inner_start = time()

        n_s: int = self.config.num_proposal_samples_per_ray[0]

        if upsample:
            ups = torch.nn.Upsample(size=(image_height, image_width, n_s), mode='nearest')
            activation_map = ups(activation_map.unsqueeze(0).unsqueeze(0)).squeeze()

        # save index arrays (coarse and fine)
        outputs = {}

        outputs['rgb'] = torch.zeros((image_height, image_width, 3)).to(self.device)
        outputs['depth'] = torch.zeros((image_height, image_width, 1)).to(self.device)
        outputs['accumulation'] = torch.zeros((image_height, image_width, 1)).to(self.device)

        num_rays: int = image_height * image_width
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            # outputs using the rays from indexed the bundle
            # re-use activation
            out = self.forward_fine(
                ray_bundle=camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx),
                act_coarse=activation_map.reshape(-1, n_s)[start_idx:end_idx],
                act_fct=act_fct)
            # overwrite based on index tuple
            outputs['rgb'].reshape(-1, 3)[start_idx:end_idx] = out['rgb']
            outputs['depth'].reshape(-1, 1)[start_idx:end_idx] = out['depth']
            outputs['accumulation'].reshape(-1, 1)[start_idx:end_idx] = out['accumulation']
        t_fine = time() - inner_start
        t_coarse = 0.

        # iterate over all index rays which fulfill the condition
        # save the metrics
        metrics = {
            't_act': t_act,
            't_coarse': t_coarse,
            't_fine': t_fine,
        }
        outputs["activation_coarse"] = activation_map.mean(-1)
        return outputs, metrics
