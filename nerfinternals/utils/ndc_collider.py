from __future__ import annotations

import torch
from nerfstudio.model_components.scene_colliders import SceneCollider
from nerfstudio.cameras.rays import RayBundle

class NDCCollider(SceneCollider):
    """Sets the nears and fars with fixed values.
    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, h: int, w: int, focal: float, **kwargs) -> None:
        self.H = h
        self.W = w
        self.focal = focal
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        near = 1.

        # Shift ray origins to near plane
        t = -(near + ray_bundle.origins[..., 2]) / ray_bundle.directions[..., 2]
        rays_o = ray_bundle.origins + t[..., None] * ray_bundle.directions

        # Projection
        o0 = -1. / (self.W / (2. * self.focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (self.H / (2. * self.focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (self.W / (2. * self.focal)) * (ray_bundle.directions[..., 0] / ray_bundle.directions[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (self.H / (2. * self.focal)) * (ray_bundle.directions[..., 1] / ray_bundle.directions[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        ray_bundle.origins = torch.stack([o0, o1, o2], -1)
        ray_bundle.directions = torch.stack([d0, d1, d2], -1)

        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        ray_bundle.nears = ones * 0.
        ray_bundle.fars = ones * 1.

        return ray_bundle