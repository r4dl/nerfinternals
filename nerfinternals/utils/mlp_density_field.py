from typing import Optional, Tuple, List, Callable

import torch
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import NeRFEncoding

from nerfinternals.utils.act_mlp import aMLP

class MLPDensityField(Field):
    """A lightweight density field module. Same basic configuration as mip-NeRF 360.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
    """

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 4,
        hidden_dim: int = 256,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )

        self.mlp_base = aMLP(
            in_dim=self.encoding.get_out_dim(),
            num_layers=self.num_layers,
            layer_width=self.hidden_dim,
            skip_connections=None,
            out_activation=None,
            out_dim=1
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)

        x = self.encoding(positions_flat).to(positions)
        density_before_activation = (
            self.mlp_base(x).view(*ray_samples.frustums.shape, -1).to(positions)
        )

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}

    def get_activations_in_layer(self, ray_samples: RaySamples, layer: int = 0) -> List[Tensor]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)

        x = self.encoding(positions_flat).to(positions)
        activation = (
            self.mlp_base.get_activation_in_layer(in_tensor=x, layer_idx=layer)
        )
        return activation.view(*ray_samples.frustums.shape, -1).mean(-1)

    def get_density_from_activation(self, positions: Tensor,
                                    layer_idx: int = 0,
                                    density_fct: Optional[Callable] = None,
                                    activation: Optional[Tensor] = None) -> Tensor:
        if activation is None:
            ray_samples = RaySamples(
                frustums=Frustums(
                    origins=positions,
                    directions=torch.ones_like(positions),
                    starts=torch.zeros_like(positions[..., :1]),
                    ends=torch.zeros_like(positions[..., :1]),
                    pixel_area=torch.ones_like(positions[..., :1]),
                )
            )
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(ray_samples.frustums.get_positions())
                positions = (positions + 2.0) / 4.0
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            # Make sure the tcnn gets inputs between 0 and 1.
            selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
            positions = positions * selector[..., None]
            positions_flat = positions.view(-1, 3)

            x = self.encoding(positions_flat).to(positions)
            # shape [b_s, N_s]
            activation = self.mlp_base.get_activation_in_layer(in_tensor=x, layer_idx=layer_idx).view(*ray_samples.frustums.shape, -1).mean(-1)

        # apply the formula to the activation
        if density_fct is not None:
            activation = density_fct(activation)
        else:
            activation = torch.relu((activation.mean(dim=-1, keepdim=True) - activation.std(dim=-1, keepdim=True)/2) - activation)**2
        # return (use this as density now, to shape [b_s, N_s, 1]
        return activation.unsqueeze(-1)