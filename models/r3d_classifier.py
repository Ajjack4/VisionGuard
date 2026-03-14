"""
VisionGuard AI — R3D-18 Violence Classifier
A fine-tunable 3D ResNet-18 backbone for binary violence classification.

SWAPPABLE: Any nn.Module accepting [B, C, T, H, W] and returning [B, 2]
can replace this model.  See pipeline/classifier.py for injection API.
"""

import torch
import torch.nn as nn


def _build_backbone(pretrained: bool = True) -> nn.Module:
    """
    Load torchvision r3d_18 with Kinetics-400 weights when available.
    Falls back to random initialisation if weights are unavailable.
    """
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights

        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            backbone = r3d_18(weights=weights)
            print("[VisionGuard] R3D-18 loaded with Kinetics-400 pretrained weights.")
        else:
            backbone = r3d_18(weights=None)
            print("[VisionGuard] R3D-18 loaded with random initialisation.")
    except (ImportError, AttributeError):
        # Older torchvision without the new Weights API
        try:
            from torchvision.models.video import r3d_18

            backbone = r3d_18(pretrained=pretrained)
            print(
                "[VisionGuard] R3D-18 loaded via legacy pretrained= API "
                "(update torchvision for the new Weights API)."
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load r3d_18 from torchvision. "
                "Install torchvision>=0.15: pip install torchvision"
            ) from exc
    return backbone


class ViolenceClassifier(nn.Module):
    """
    Binary violence classifier built on top of R3D-18.

    Input  : Tensor [B, C, T, H, W]  (B=batch, C=3, T=16, H=W=112)
    Output : Tensor [B, 2]           softmax probabilities
                                     [p_nonviolent, p_violent]

    The model is intentionally easy to swap:
      - Train your own backbone externally (MMAction2 / PySlowFast / etc.)
      - Save as torch.save(model.state_dict(), 'my_model.pth')
      - Point MODEL_PATH in .env to your file — classifier.py loads it here.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        backbone = _build_backbone(pretrained=pretrained)

        # Replace the original Kinetics-400 head (512 → 400) with a binary head
        in_features: int = backbone.fc.in_features  # 512 for r3d_18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor [B, 3, T, H, W], values in [0, 1] after
               normalisation (mean/std applied in detector crop stage).
        Returns:
            Softmax probabilities [B, 2]:  [:, 0] = non-violent, [:, 1] = violent.
        """
        logits = self.backbone(x)  # [B, 2]
        return torch.softmax(logits, dim=1)


# ── Convenience factory ─────────────────────────────────────────────────────

def build_model(
    pretrained: bool = True,
    dropout: float = 0.5,
    state_dict_path: str | None = None,
    device: str = "cpu",
) -> ViolenceClassifier:
    """
    Factory function to construct and optionally load a fine-tuned model.

    Args:
        pretrained: Use Kinetics-400 weights as backbone init.
        dropout: Dropout rate before the final FC layer.
        state_dict_path: Optional path to a fine-tuned .pth state dict.
        device: Target device string.

    Returns:
        ViolenceClassifier ready for inference or further fine-tuning.
    """
    model = ViolenceClassifier(pretrained=pretrained, dropout=dropout)

    if state_dict_path is not None:
        import pathlib

        p = pathlib.Path(state_dict_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model state dict not found: {state_dict_path}\n"
                "Check MODEL_PATH in your .env file."
            )
        state = torch.load(str(p), map_location=device)
        # Support both plain state dicts and checkpoint dicts
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"[VisionGuard] Fine-tuned weights loaded from: {p}")

    model.to(device)
    return model
