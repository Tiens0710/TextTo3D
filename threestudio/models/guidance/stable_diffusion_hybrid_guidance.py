import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.guidance.stable_diffusion_sdi_guidance import (
    StableDiffusionSDIGuidance,
)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.misc import C
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-hybrid-guidance")
class StableDiffusionHybridGuidance(StableDiffusionSDIGuidance):
    """
    Hybrid SDS+SDI Guidance for Text-to-3D.

    Strategy:
      - Phase 1 (step < sdi_start_step):         Pure SDS  → fast geometry formation
      - Phase 2 (sdi_start_step ≤ step < start+warmup): Blend   → smooth transition
      - Phase 3 (step ≥ sdi_start_step + warmup): Pure SDI  → sharp texture refinement

    Also integrates adaptive guidance scale scheduling:
      - guidance_scale linearly decays from guidance_scale_start to guidance_scale_end

    Inherits from StableDiffusionSDIGuidance which already has both SDS and SDI
    capability (enable_sdi=True for SDI, enable_sdi=False or random noise for SDS).
    """

    @dataclass
    class Config(StableDiffusionSDIGuidance.Config):
        # Phase transition config
        sdi_start_step: int = 3000  # step where SDI begins
        sdi_warmup_steps: int = 1000  # steps for smooth SDS→SDI blend

        # Adaptive guidance scale
        guidance_scale_start: float = 100.0
        guidance_scale_end: float = 50.0
        guidance_scale_warmup_steps: int = 500

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self._current_phase = "sds"  # track for logging
        threestudio.info(
            f"[HybridGuidance] SDS phase: 0→{self.cfg.sdi_start_step}, "
            f"Blend: {self.cfg.sdi_start_step}→{self.cfg.sdi_start_step + self.cfg.sdi_warmup_steps}, "
            f"SDI phase: {self.cfg.sdi_start_step + self.cfg.sdi_warmup_steps}+, "
            f"guidance_scale: {self.cfg.guidance_scale_start}→{self.cfg.guidance_scale_end}"
        )

    def _get_blend_alpha(self, global_step: int) -> float:
        """
        Returns alpha in [0, 1] for blending SDS→SDI.
        alpha=0 → pure SDS, alpha=1 → pure SDI
        """
        if global_step < self.cfg.sdi_start_step:
            return 0.0
        elif global_step >= self.cfg.sdi_start_step + self.cfg.sdi_warmup_steps:
            return 1.0
        else:
            return (global_step - self.cfg.sdi_start_step) / max(
                1, self.cfg.sdi_warmup_steps
            )

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        global_step = kwargs.get("global_step", 0)

        alpha = self._get_blend_alpha(global_step)

        if alpha == 0.0:
            # Phase 1: Pure SDS (use random noise instead of DDIM inversion)
            self._current_phase = "sds"
            random_noise = torch.randn(batch_size, 4, 64, 64, device=self.device)
            return super().__call__(
                rgb=rgb,
                prompt_utils=prompt_utils,
                elevation=elevation,
                azimuth=azimuth,
                camera_distances=camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=guidance_eval,
                call_with_defined_noise=random_noise,
                **{k: v for k, v in kwargs.items() if k != "global_step"},
            )
        elif alpha == 1.0:
            # Phase 3: Pure SDI (use DDIM inversion)
            self._current_phase = "sdi"
            return super().__call__(
                rgb=rgb,
                prompt_utils=prompt_utils,
                elevation=elevation,
                azimuth=azimuth,
                camera_distances=camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=guidance_eval,
                **{k: v for k, v in kwargs.items() if k != "global_step"},
            )
        else:
            # Phase 2: Blend SDS + SDI
            self._current_phase = "blend"

            # Compute SDS loss (with random noise)
            random_noise = torch.randn(batch_size, 4, 64, 64, device=self.device)
            sds_out = super().__call__(
                rgb=rgb,
                prompt_utils=prompt_utils,
                elevation=elevation,
                azimuth=azimuth,
                camera_distances=camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=False,
                call_with_defined_noise=random_noise,
                **{k: v for k, v in kwargs.items() if k != "global_step"},
            )

            # Compute SDI loss (with DDIM inversion)
            sdi_out = super().__call__(
                rgb=rgb,
                prompt_utils=prompt_utils,
                elevation=elevation,
                azimuth=azimuth,
                camera_distances=camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=guidance_eval,
                **{k: v for k, v in kwargs.items() if k != "global_step"},
            )

            # Blend losses
            blended_loss = (1 - alpha) * sds_out["loss_sdi"] + alpha * sdi_out[
                "loss_sdi"
            ]

            guidance_out = {
                "loss_sdi": blended_loss,
                "grad_norm": (1 - alpha) * sds_out["grad_norm"]
                + alpha * sdi_out["grad_norm"],
                "min_step": sdi_out["min_step"],
                "max_step": sdi_out["max_step"],
                "loss_sds_component": sds_out["loss_sdi"],
                "loss_sdi_component": sdi_out["loss_sdi"],
                "blend_alpha": alpha,
            }

            if guidance_eval and "eval" in sdi_out:
                guidance_out["eval"] = sdi_out["eval"]

            return guidance_out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        # --- Adaptive guidance scale ---
        if global_step < self.cfg.guidance_scale_warmup_steps:
            self.cfg.guidance_scale = self.cfg.guidance_scale_start
        else:
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

        # --- Timestep annealing (from parent SDI) ---
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.t_anneal:
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
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )

    @property
    def current_phase(self) -> str:
        return self._current_phase
