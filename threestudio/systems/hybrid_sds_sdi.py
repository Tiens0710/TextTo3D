from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.sdi import ScoreDistillationViaInversion
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("hybrid-sds-sdi-system")
class HybridSDSSDISystem(ScoreDistillationViaInversion):
    """
    Training system for Hybrid SDS+SDI guidance.

    Inherits from ScoreDistillationViaInversion (SDI system) which already
    handles both SDS and SDI losses. This system adds:
    - Passing global_step to guidance for phase-aware computation
    - Logging of hybrid-specific metrics (phase, blend_alpha, component losses)
    """

    @dataclass
    class Config(ScoreDistillationViaInversion.Config):
        pass

    cfg: Config

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()

        # Pass global_step so hybrid guidance knows which phase to use
        guidance_out = self.guidance(
            out["comp_rgb"],
            prompt_utils,
            **batch,
            rgb_as_latents=False,
            global_step=self.true_global_step,
        )

        loss = 0.0

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss.get(
                        name.replace("loss_", "lambda_"),
                        1.0,  # default lambda=1.0 for new loss keys
                    )
                )

        # Log hybrid phase info
        if hasattr(self.guidance, "current_phase"):
            phase_map = {"sds": 0, "blend": 1, "sdi": 2}
            self.log(
                "train/hybrid_phase",
                phase_map.get(self.guidance.current_phase, -1),
            )

        if hasattr(self.guidance, "cfg"):
            self.log("train/guidance_scale", self.guidance.cfg.guidance_scale)

        # Orientation loss
        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        # Sparsity loss
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        # Opaque loss
        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z-variance loss (HiFA)
        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
