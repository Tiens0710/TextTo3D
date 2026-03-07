import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import (
    StableDiffusionPromptProcessor,
)
from threestudio.utils.typing import *


@threestudio.register("enhanced-sd-prompt-processor")
class EnhancedSDPromptProcessor(StableDiffusionPromptProcessor):
    """
    Enhanced Stable Diffusion Prompt Processor with stronger Anti-Janus measures.

    The Janus problem occurs when the 3D model shows the same face/front-view
    features on multiple sides (especially the back). This processor addresses
    it by:
    1. Adding extra negative prompts for back and side views
    2. Scaling the negative embedding strength for back views
    3. Works with both standard view-dependent prompting and Perp-Neg

    Usage in config:
        prompt_processor_type: "enhanced-sd-prompt-processor"
        prompt_processor:
            enhanced_back_negative: true
            back_negative_extra: "face, eyes, mouth, front view, looking at viewer"
            side_negative_extra: "front view, face"
            back_negative_scale: 2.0
    """

    @dataclass
    class Config(StableDiffusionPromptProcessor.Config):
        # Enhanced anti-Janus settings
        enhanced_back_negative: bool = True
        back_negative_extra: str = (
            "face, eyes, mouth, front view, looking at viewer"
        )
        side_negative_extra: str = "front view, face"
        back_negative_scale: float = 2.0  # scale factor for back-view negative embedding

    cfg: Config

    def configure(self) -> None:
        # Call parent configure which sets up all prompts and embeddings
        super().configure()

        if self.cfg.enhanced_back_negative:
            threestudio.info(
                f"[EnhancedAntiJanus] Applying enhanced negative prompts: "
                f"back='{self.cfg.back_negative_extra}', "
                f"side='{self.cfg.side_negative_extra}', "
                f"back_scale={self.cfg.back_negative_scale}"
            )

            # Enhance negative prompts for back/side views
            # Direction indices: 0=side, 1=front, 2=back, 3=overhead
            original_back_neg = self.negative_prompts_vd[2]
            original_side_neg = self.negative_prompts_vd[0]

            # Append extra negative terms
            if self.cfg.back_negative_extra:
                enhanced_back_neg = (
                    f"{original_back_neg}, {self.cfg.back_negative_extra}"
                    if original_back_neg
                    else self.cfg.back_negative_extra
                )
                self.negative_prompts_vd[2] = enhanced_back_neg

            if self.cfg.side_negative_extra:
                enhanced_side_neg = (
                    f"{original_side_neg}, {self.cfg.side_negative_extra}"
                    if original_side_neg
                    else self.cfg.side_negative_extra
                )
                self.negative_prompts_vd[0] = enhanced_side_neg

            threestudio.info(
                f"[EnhancedAntiJanus] Updated negative prompts: "
                f"side='{self.negative_prompts_vd[0]}', "
                f"back='{self.negative_prompts_vd[2]}'"
            )

            # Re-prepare and reload text embeddings with enhanced negatives
            self.prepare_text_embeddings()
            self.load_text_embeddings()

            # Scale the back-view negative embedding for stronger penalty
            if self.cfg.back_negative_scale != 1.0:
                self.uncond_text_embeddings_vd[2] = (
                    self.uncond_text_embeddings_vd[2] * self.cfg.back_negative_scale
                )
                threestudio.info(
                    f"[EnhancedAntiJanus] Scaled back-view negative embedding by {self.cfg.back_negative_scale}x"
                )
