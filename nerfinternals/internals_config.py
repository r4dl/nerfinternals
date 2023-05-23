from __future__ import annotations

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
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
            model=NeRFInternalModelConfig(_target=MipNerfInternalModel),
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
    description="Using Activations to infer Depth, Mip-NeRF.",
)