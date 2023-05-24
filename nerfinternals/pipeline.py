"""
Custom Pipeline for our experiments.
"""
from __future__ import annotations

import json
import typing, os
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Type, Union, cast, Tuple, Callable
from torchvision.utils import save_image

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import Pipeline

from nerfinternals.eval import ActivationDerivedDensity


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


@dataclass
class InternalsPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: InternalVanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class InternalVanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: InternalVanillaPipeline,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # example camera
        c = self.datamanager.train_dataset.cameras[0]

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            hwf=[c.height.item(), c.width.item(), c.fx.item()]
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}

    def create_overarching_dir(self, directory: str, folder_name: str, idx: str) -> str:
        assert os.path.exists(directory)
        img_directory = os.path.join(directory, folder_name)
        if not os.path.exists(img_directory):
            os.mkdir(img_directory)
        # create the per-img subdirectory
        path_to_images = os.path.join(img_directory, idx)
        if not os.path.exists(path_to_images):
            os.mkdir(path_to_images)
        return path_to_images

    @profiler.time_function
    def activation_derived_density_NeRF(self, save_dir: str, options: ActivationDerivedDensity):
        """Use activations as proxy for where importance weighted samples are needed

        Args:
            save_dir: where to save the output
            options:
        """
        self.eval()

        # the three functions we propose
        activation_functions: List[Tuple[str, Callable]] = [
            ('std', lambda x: torch.relu((x.mean(-1, keepdim=True) - x.std(-1, keepdim=True)) - x)),
            ('std_half', lambda x: torch.relu((x.mean(-1, keepdim=True) - x.std(-1, keepdim=True) / 2) - x)),
            ('std_half_sq', lambda x: torch.relu((x.mean(-1, keepdim=True) - x.std(-1, keepdim=True) / 2) - x) ** 2),
        ]
        run_normal: bool = options.run_normal

        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            # create a folder for each image
            path_to_images_: str = self.create_overarching_dir(
                directory=save_dir,
                folder_name=options.output_dir,
                idx=f'{batch["image_idx"]:03}')

            # first: get outputs & metrics normally
            inner_start = time()
            outputs_normal, metrics_normal = None, None
            if run_normal:
                outputs_normal = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            t_normal = time() - inner_start

            metrics_all = {}
            if outputs_normal is not None:
                metrics_normal, _ = self.model.get_image_metrics_and_images(outputs_normal, batch)
                # render image & target
                save_image(outputs_normal["rgb"].permute(-1, 0, 1), os.path.join(path_to_images_, f"coarse-to-fine.png"))
                save_image(batch["image"].permute(-1, 0, 1), os.path.join(path_to_images_, f"target.png"))
                # save memory
                del outputs_normal

                metrics_all["base"] = {
                    "t": t_normal,
                    "metrics": metrics_normal
                }

            # num_samples for either NeRF, Mip-NeRF or nerfacto
            n_s: int = self.config.model.num_proposal_samples_per_ray[0] if self.config.model.__class__.__name__ in 'NerfactoInternalModelConfig' else self.config.model.num_coarse_samples

            # downsample by a factor of 2
            upsample_resolution: List[int] = [b // 2 for b in batch['image'].shape[:2]] + [n_s//2]
            upsample = options.upsample
            for layer in list(options.layer):
                for fct_idx in options.fct:
                    run_str, run_fct = activation_functions[fct_idx]
                    exp_name: str = f'layer_{layer:02}_ups_{int(upsample)}_fct_{run_str}'
                    path_to_images = os.path.join(path_to_images_, exp_name)
                    if not os.path.exists(path_to_images):
                        os.mkdir(path_to_images)
                    # second: get outputs optimized
                    inner_start = time()
                    outputs_optimized, quant_metr = self.model.get_outputs_for_camera_ray_bundle_activationinformed(
                        camera_ray_bundle,
                        layer=layer,
                        num_samples=n_s,
                        upsample=upsample,
                        upsample_res=upsample_resolution,
                        act_fct=run_fct)
                    t_optimized = time() - inner_start

                    save_image(outputs_optimized["rgb"].permute(-1, 0, 1),
                               os.path.join(path_to_images, f"ours.png"))

                    save_image(batch['image'].permute((-1, 0, 1)), os.path.join(path_to_images, f'target.png'))

                    metrics_optimized, _ = self.model.get_image_metrics_and_images(outputs_optimized, batch)
                    metrics_optimized["quantitative"] = quant_metr

                    # save normal pipeline metrics into dict
                    metrics = {}
                    metrics_time = {
                        "t_normal": t_normal,
                        "t_optimized": t_optimized,
                        "percentage": t_optimized / t_normal,
                    }
                    if metrics_normal is not None:
                        metrics["normal"] = metrics_normal
                    metrics["optimized"] = metrics_optimized
                    metrics["time"] = metrics_time

                    metrics_all[exp_name] = {
                        "t": t_optimized,
                        "metrics": metrics_optimized
                    }
                    print(f'{metrics_optimized["psnr"]:.3f}, layer_{layer} ups_{int(upsample)}, {run_str}')

                    # write to json
                    with open(os.path.join(path_to_images, 'stats.json'), "w") as outfile:
                        outfile.write(json.dumps(metrics, indent=2))
            with open(os.path.join(path_to_images_, 'stats.json'), "w") as outfile:
                outfile.write(json.dumps(metrics_all, indent=2))
            print(f'finished image {batch["image_idx"]:03}')
