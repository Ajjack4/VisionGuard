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
        import config

        # Injected nn.Module — use directly regardless of MODEL_TYPE
        if isinstance(source, nn.Module):
            return source.to(self.device)

        model_type = config.MODEL_TYPE

        # ── kinetics_heuristic: zero-shot, works immediately ──────────────────
        if model_type == "kinetics_heuristic":
            from models.kinetics_heuristic import build_heuristic_model
            return build_heuristic_model(
                backbone="r3d_18",
                score_scale=config.KINETICS_SCORE_SCALE,
                device=self.device,
            )

        # ── X3D / SlowFast via torch.hub ──────────────────────────────────────
        if model_type in ("x3d_xs", "x3d_s", "x3d_m", "slowfast_r50"):
            from models.kinetics_heuristic import build_heuristic_model
            return build_heuristic_model(
                backbone=model_type,
                score_scale=config.KINETICS_SCORE_SCALE,
                device=self.device,
            )

        # ── r3d18: binary head, requires fine-tuning to be useful ─────────────
        if model_type == "r3d18":
            from models.r3d_classifier import build_model
            state_dict = source if isinstance(source, str) else config.MODEL_PATH
            return build_model(
                pretrained=True,
                state_dict_path=state_dict,
                device=self.device,
            )

        # ── slowfast_violence: fine-tuned SlowFast R50 (train_slowfast.ipynb) ──
        if model_type == "slowfast_violence":
            import json
            from models.slowfast_wrapper import SlowFastWrapper
            model_path = source if isinstance(source, str) else config.MODEL_PATH
            meta_path  = (model_path or "").replace(".pt", ".json")
            alpha      = 4  # default; overridden by saved metadata if present
            try:
                with open(meta_path) as f:
                    meta  = json.load(f)
                    alpha = meta.get("alpha", alpha)
            except (FileNotFoundError, ValueError):
                pass
            wrapper = SlowFastWrapper(alpha=alpha, num_classes=2)
            if model_path:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                wrapper.load_state_dict(state)
                print(f"[Classifier] Loaded slowfast_violence weights from {model_path}")
            else:
                print("[Classifier] WARNING: MODEL_PATH not set — using untrained SlowFast weights")
            return wrapper.to(self.device)

        # Unknown — fall back with warning
        print(
            f"[Classifier] WARNING: unknown MODEL_TYPE='{model_type}'. "
            "Falling back to kinetics_heuristic."
        )
        from models.kinetics_heuristic import build_heuristic_model
        return build_heuristic_model(
            backbone="r3d_18",
            score_scale=config.KINETICS_SCORE_SCALE,
            device=self.device,
        )

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
