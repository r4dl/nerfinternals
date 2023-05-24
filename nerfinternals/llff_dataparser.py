"""Data parser for LLFF dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional
import os

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


@dataclass
class LLFFDataParserConfig(DataParserConfig):
    """LLFF dataset parser config"""

    _target: Type = field(default_factory=lambda: LLFF)
    """target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 4
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""

@dataclass
class LLFF(DataParser):
    """LLFF Dataset parser
    Most of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37
    and is adapted to fit Nerfstudio
    """

    config: LLFFDataParserConfig

    def __init__(self, config: LLFFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.downscale_factor: Optional[int] = config.downscale_factor

    def _generate_dataparser_outputs(self, split="train"):

        # Code directly taken from nerf-pytorch
        # https://github.com/yenchenlin/nerf-pytorch
        def normalize(x):
            return x / np.linalg.norm(x)

        def viewmatrix(z, up, pos):
            vec2 = normalize(z)
            vec1_avg = up
            vec0 = normalize(np.cross(vec1_avg, vec2))
            vec1 = normalize(np.cross(vec2, vec0))
            m = np.stack([vec0, vec1, vec2, pos], 1)
            return m

        def poses_avg(poses):

            hwf = poses[0, :3, -1:]

            center = poses[:, :3, 3].mean(0)
            vec2 = normalize(poses[:, :3, 2].sum(0))
            up = poses[:, :3, 1].sum(0)
            c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

            return c2w
        def recenter_poses(poses):

            poses_ = poses + 0
            bottom = np.reshape([0, 0, 0, 1.], [1, 4])
            c2w = poses_avg(poses)
            c2w = np.concatenate([c2w[:3, :4], bottom], -2)
            bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
            poses = np.concatenate([poses[:, :3, :4], bottom], -2)

            poses = np.linalg.inv(c2w) @ poses
            poses_[:, :3, :4] = poses[:, :3, :4]
            poses = poses_
            return poses

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        # load data according to _load_data in nerf-pytorch
        poses_arr = np.load(os.path.join(self.config.data, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])

        # handle different scales
        if self.downscale_factor is not None:
            imgdir = os.path.join(self.config.data, f'images_{self.downscale_factor}')
        else:
            imgdir = os.path.join(self.config.data, f'../images')
        image_filenames = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                        f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        # load example image
        sh = imageio.imread(image_filenames[0]).shape
        # reshape poses
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.downscale_factor

        # sanity check print
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        sc = 1. if self.scale_factor is None else 1. / (bds.min() * self.scale_factor)
        # sc (above) should be 0.078 for fern at 8
        poses[:, :3, 3] *= sc
        bds *= sc

        # recenter poses (all the same up to this point)
        poses = recenter_poses(poses)

        # next: things from train() in nerf-pytorch
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        camera_to_world = torch.from_numpy(poses)

        # handle split
        num_images = len(image_filenames)
        num_train_images = int(np.ceil(num_images * self.config.train_split_fraction))
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # use only the poses, images according to split
        image_filenames = [image_filenames[i] for i in indices]
        camera_to_world = camera_to_world[indices]

        # From this point onwards, this is equivalent to the blender dataloader
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal.item(),
            fy=focal.item(),
            cx=W / 2.0,
            cy=H / 2.0,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
