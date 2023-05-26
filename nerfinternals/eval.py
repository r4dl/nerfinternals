#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import os
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)

@dataclass
class ActivationDerivedDensity:
    """Load a checkpoint, use the activations for estimating the density."""

    # Path to config YAML file.
    load_config: Path
    # layer in which to observe the activations - must not be larger than num_layers
    layer: Tuple[int, ...] = (0, 1, 2)
    # function to use - must not be larger than 2
    fct: Tuple[int, ...] = (0, 1, 2)
    # whether to upsample or not
    upsample: bool = False
    # whether to run coarse-to-fine pipeline or not
    run_normal: bool = True
    # directory to save outputs in
    output_dir: str = "eval"

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        pipeline.activation_derived_density_NeRF(save_dir=os.path.dirname(config.load_dir),
                                                 options=self)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ActivationDerivedDensity).main()


if __name__ == "__main__":
    entrypoint()