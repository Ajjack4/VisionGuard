"""
VisionGuard AI — SlowFast R50 Wrapper
Fine-tuned binary violence classifier compatible with ClipClassifier.

Accepts standard [B, C, T, H, W] float32 input (T = BUFFER_SIZE, e.g. 32).
Internally splits into Slow (T//alpha) and Fast (T) pathways.
Returns softmax probabilities [B, 2]: index 0 = non-violent, 1 = violent.

Usage
-----
After training with train_slowfast.ipynb:
    MODEL_TYPE=slowfast_violence
    MODEL_PATH=models/slowfast_violence.pt
    BUFFER_SIZE=32
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlowFastWrapper(nn.Module):
    """
    Wraps PyTorchVideo SlowFast R50 for binary violence classification.

    Parameters
    ----------
    alpha      : int   Slow/Fast temporal ratio (Slow = T // alpha frames).
    num_classes: int   Output classes (2 for violence / non-violence).
    pretrained : bool  Load Kinetics-400 weights from torch.hub on init.
    """

    def __init__(
        self,
        alpha: int = 4,
        num_classes: int = 2,
        pretrained: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=pretrained,
        )
        # Replace Kinetics head with binary head
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, T_fast, H, W]  (T_fast = BUFFER_SIZE, e.g. 32)

        Returns
        -------
        [B, 2]  softmax probabilities
        """
        slow = x[:, :, :: self.alpha, :, :]   # [B, C, T_slow, H, W]
        logits = self.backbone([slow, x])
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final projection head (Phase 1 training)."""
        for name, param in self.backbone.named_parameters():
            if "blocks.5.proj" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (Phase 2 full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
