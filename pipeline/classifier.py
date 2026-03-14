"""
VisionGuard AI — Clip Classifier
Wraps the 3D-CNN model for inference.  Supports model injection and hot-swap.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class ClipClassifier:
    """
    Runs the violence classifier on pre-built clip tensors.

    The classifier is SWAPPABLE: pass either
    * a string path   — loaded as a state dict onto ViolenceClassifier, or
    * an nn.Module    — used directly (inject any compatible model).

    Any model that accepts [B, C, T, H, W] float32 and returns [B, 2]
    (softmax probabilities) is compatible.

    Parameters
    ----------
    model_path_or_module : str | nn.Module | None
        Path to .pth state dict, a ready nn.Module, or None for default init.
    device : str
        "cuda", "mps", or "cpu".
    threshold : float
        P(violent) threshold above which a clip is flagged positive.
    """

    def __init__(
        self,
        model_path_or_module: Union[str, nn.Module, None] = None,
        device: str = "cpu",
        threshold: float = 0.65,
    ):
        self.device = device
        self.threshold = threshold
        self._model: nn.Module = self._load_model(model_path_or_module)
        self._model.eval()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self, source: Union[str, nn.Module, None]) -> nn.Module:
        from models.r3d_classifier import build_model

        if source is None:
            # Default: Kinetics-pretrained backbone, no fine-tuning
            model = build_model(
                pretrained=True,
                device=self.device,
            )
        elif isinstance(source, str):
            model = build_model(
                pretrained=True,
                state_dict_path=source,
                device=self.device,
            )
        elif isinstance(source, nn.Module):
            model = source.to(self.device)
        else:
            raise TypeError(
                f"model_path_or_module must be str, nn.Module, or None. "
                f"Got: {type(source)}"
            )
        return model

    def swap_model(self, new_model: nn.Module) -> None:
        """Hot-swap the underlying model at runtime (e.g. after fine-tuning)."""
        self._model = new_model.to(self.device)
        self._model.eval()
        print("[Classifier] Model swapped successfully.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, clip: np.ndarray) -> float:
        """
        Classify a single temporal clip.

        Parameters
        ----------
        clip : np.ndarray  [C, T, H, W]  float32

        Returns
        -------
        float — P(violent) in [0, 1]
        """
        tensor = torch.from_numpy(clip).unsqueeze(0)  # [1, C, T, H, W]
        return self.predict_batch([clip])[0]

    def predict_batch(self, clips: List[np.ndarray]) -> List[float]:
        """
        Classify a batch of clips.

        Parameters
        ----------
        clips : List[np.ndarray]  each [C, T, H, W]  float32

        Returns
        -------
        List[float]  P(violent) for each clip.
        """
        batch = torch.stack(
            [torch.from_numpy(c) for c in clips], dim=0
        ).to(self.device)  # [B, C, T, H, W]

        with torch.no_grad():
            probs = self._model(batch)  # [B, 2]

        p_violent = probs[:, 1].cpu().numpy().tolist()
        return p_violent

    def is_violent(self, confidence: float) -> bool:
        """Return True if P(violent) exceeds the configured threshold."""
        return confidence >= self.threshold
