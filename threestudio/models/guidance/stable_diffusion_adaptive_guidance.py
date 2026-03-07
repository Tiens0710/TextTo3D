from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import (
    StableDiffusionGuidance,
)
from threestudio.utils.misc import C
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-adaptive-guidance")
class StableDiffusionAdaptiveGuidance(StableDiffusionGuidance):
    """
    Adaptive Guidance Scale Scheduling for Stable Diffusion SDS.

    Key idea: Start with high guidance_scale (strong shape formation),
    gradually decrease it (more natural textures, less over-saturation).

    Also integrates linear timestep annealing (complementary to sqrt_anneal).
    """

    @dataclass
    class Config(StableDiffusionGuidance.Config):
        # Adaptive guidance scale scheduling
        guidance_scale_start: float = 100.0
        guidance_scale_end: float = 30.0
        guidance_scale_warmup_steps: int = 1000  # keep high guidance during warmup

        # Linear timestep annealing (alternative to sqrt_anneal)
        use_linear_t_anneal: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()
        # Override the fixed guidance_scale with the start value
        self.cfg.guidance_scale = self.cfg.guidance_scale_start
        threestudio.info(
            f"[AdaptiveGuidance] guidance_scale: {self.cfg.guidance_scale_start} → {self.cfg.guidance_scale_end}, "
            f"warmup: {self.cfg.guidance_scale_warmup_steps} steps"
        )

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        # --- Adaptive guidance scale ---
        if global_step < self.cfg.guidance_scale_warmup_steps:
            # Keep high guidance during warmup (shape formation phase)
            self.cfg.guidance_scale = self.cfg.guidance_scale_start
        else:
            # Linear decay from start to end
            progress = min(
                1.0,
                (global_step - self.cfg.guidance_scale_warmup_steps)
                / max(
                    1,
                    self.cfg.trainer_max_steps - self.cfg.guidance_scale_warmup_steps,
                ),
            )
            self.cfg.guidance_scale = (
                self.cfg.guidance_scale_start
                + (self.cfg.guidance_scale_end - self.cfg.guidance_scale_start)
                * progress
            )

        # --- Timestep annealing ---
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.use_linear_t_anneal and not self.cfg.sqrt_anneal:
            # Linear annealing: max_step_percent decreases linearly
            percentage = float(global_step) / self.cfg.trainer_max_steps
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        elif self.cfg.sqrt_anneal:
            # Use parent's sqrt anneal
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
