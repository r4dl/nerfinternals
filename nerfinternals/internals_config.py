from __future__ import annotations


from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfinternals.model import NeRFInternalModelConfig, NeRFInternalModel
from nerfinternals.pipeline import InternalsPipelineConfig, InternalVanillaPipeline

activation_nerf_blender = TrainerConfig(
    method_name="activation-nerf",
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
)