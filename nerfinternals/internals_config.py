from __future__ import annotations

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfinternals.nerf_model import NeRFInternalModelConfig, NeRFInternalModel
from nerfinternals.mipnerf_model import MipNerfInternalModel
from nerfinternals.pipeline import InternalsPipelineConfig, InternalVanillaPipeline

activation_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="activation-nerf",
        steps_per_eval_image=1000001,
        steps_per_eval_batch=1000001,
        steps_per_eval_all_images=1000001,
        pipeline=InternalsPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
            ),
            model=NeRFInternalModelConfig(_target=NeRFInternalModel),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        vis="tensorboard",
    ),
    description="Using Activations to infer Depth, NeRF.",
)

activation_mipnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="activation-mipnerf",
        steps_per_eval_image=1000001,
        steps_per_eval_batch=1000001,
        steps_per_eval_all_images=1000001,
        pipeline=InternalsPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
            ),
            model=NeRFInternalModelConfig(
                _target=MipNerfInternalModel,
                loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
                num_coarse_samples=128,
                num_importance_samples=128,
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            }
        },
        vis="tensorboard",
    ),
    description="Using Activations to infer Depth, Mip-NeRF.",
)