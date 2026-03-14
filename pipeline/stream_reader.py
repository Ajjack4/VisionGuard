"""
VisionGuard AI — Stream Reader
Handles webcam, IP Webcam (Android), RTSP, file, and YouTube live sources.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def _resolve_youtube(url: str, quality: str = "best[height<=720]") -> str:
    """
    Use yt-dlp to extract the direct stream URL from a YouTube URL.

    Works for both live streams and regular videos.
    Falls back to 'best' format if the quality selector yields nothing.

    Requires: pip install yt-dlp
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube sources.\n"
            "Install it with:  pip install yt-dlp"
        )

    ydl_opts = {
        "format": quality,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    print(f"[StreamReader] Resolving YouTube URL (quality={quality}) …")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError:
            # Retry with unrestricted format
            print("[StreamReader] Quality selector failed, retrying with 'best' …")
            ydl_opts["format"] = "best"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                info = ydl2.extract_info(url, download=False)

    # For playlists/live streams the direct URL lives in 'url'
    direct_url: str = info.get("url") or info["formats"][-1]["url"]
    title: str = info.get("title", "YouTube stream")
    print(f"[StreamReader] Resolved: '{title}'")
    return direct_url


def _is_youtube_url(s: str) -> bool:
    return any(
        domain in s
        for domain in ("youtube.com/", "youtu.be/", "youtube.com/live")
    )


class StreamReader:
    """
    Unified video source reader.

    Accepted sources
    ----------------
    * int / "0"                     — webcam index
    * "http://IP:8080/video"        — Android IP Webcam app
    * "rtsp://…"                    — RTSP IP camera
    * "/path/to/file.mp4"           — local video file
    * "https://youtube.com/…"       — YouTube video or live stream (via yt-dlp)
    """

    def __init__(self, source: Union[int, str], youtube_quality: str = "best[height<=720]"):
        self._original_source = source
        self._youtube_quality = youtube_quality
        self._cap: cv2.VideoCapture | None = None
        self._connected: bool = False

        # Resolve YouTube URLs to a direct stream URL before opening
        if isinstance(source, str) and _is_youtube_url(source):
            self.source = _resolve_youtube(source, youtube_quality)
            self._source_label = f"YouTube ({source})"
        else:
            self.source = source
            self._source_label = self._make_label(source)

    # ── Public API ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the video source.  Returns True on success."""
        self._cap = cv2.VideoCapture(self.source)
        # Give USB / network sources a moment to negotiate
        time.sleep(0.3)
        self._connected = self._cap.isOpened()
        if not self._connected:
            print(
                f"[StreamReader] ERROR: could not open source: {self.source}\n"
                "  • For webcam, try source 0 or 1.\n"
                "  • For IP Webcam, make sure your phone and PC are on the "
                "same Wi-Fi network and the app is running."
            )
        else:
            label = self._source_label
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            print(
                f"[StreamReader] Connected — {label}  "
                f"({w}x{h} @ {fps:.1f} fps)"
            )
        return self._connected

    def read_frame(self) -> np.ndarray | None:
        """
        Read the next frame.

        Returns an RGB numpy array [H, W, 3] on success, or None on failure.
        The frame is converted from BGR to RGB so downstream code always
        receives RGB.  (OpenCV draw calls in main.py will re-convert where
        needed.)
        """
        if self._cap is None or not self._connected:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._connected = False
            return None
        # Return BGR — keep consistent with OpenCV everywhere in the pipeline
        return frame

    def release(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_label(self) -> str:
        return self._source_label

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_label(source: Union[int, str]) -> str:
        if isinstance(source, int):
            return f"Webcam #{source}"
        s = str(source)
        if s.isdigit():
            return f"Webcam #{s}"
        if _is_youtube_url(s):
            return f"YouTube ({s})"
        if s.startswith("rtsp://"):
            return "RTSP Camera"
        if "/video" in s or "IP_WEBCAM" in s.upper():
            return f"IP Webcam ({s})"
        if Path(s).exists():
            return f"File: {Path(s).name}"
        return f"Stream: {s}"

    @staticmethod
    def is_ip_webcam_url(source: str) -> bool:
        """Return True if the URL looks like an Android IP Webcam feed."""
        return "/video" in source or "IP_WEBCAM" in source.upper()


# ── Mobile camera setup instructions ────────────────────────────────────────

def get_mobile_camera_instructions() -> str:
    """
    Print step-by-step instructions for connecting a mobile phone as a camera.
    """
    return """
╔══════════════════════════════════════════════════════════════════════════╗
║              MOBILE CAMERA SETUP INSTRUCTIONS                          ║
╚══════════════════════════════════════════════════════════════════════════╝

── ANDROID (IP Webcam app) ──────────────────────────────────────────────
  1. Install "IP Webcam" by Pavel Khlebovich from the Google Play Store.
  2. Open the app and scroll to the bottom — tap "Start server".
  3. Your phone's IP address and port are displayed on screen, e.g.
         http://192.168.1.42:8080
  4. On your PC, make sure both devices are on the SAME Wi-Fi network.
  5. In your browser, open http://192.168.1.42:8080 to verify the feed.
  6. Set SOURCE in .env (or pass --source flag):
         SOURCE=http://192.168.1.42:8080/video
  7. Run:  python main.py

── iPHONE — Option A: Continuity Camera (macOS 13+ / iOS 16+) ──────────
  1. Make sure your iPhone and Mac are signed in to the same Apple ID.
  2. In macOS System Settings → General → AirPlay & Handoff, enable
     "Continuity Camera".
  3. On your iPhone go to Settings → General → AirPlay & Handoff and
     enable "Continuity Camera".
  4. Open any app that uses the camera on macOS — your iPhone should
     appear automatically as a camera option (no app needed on iPhone).
  5. VisionGuard will detect it as an additional webcam index (try 1, 2…):
         SOURCE=1

── iPHONE — Option B: EpocCam ──────────────────────────────────────────
  1. Install "EpocCam Webcam" from the App Store (free tier available).
  2. Install the EpocCam driver on your PC/Mac from the Kinoni website.
  3. Connect iPhone and computer to the SAME Wi-Fi (or via USB).
  4. Open the EpocCam app on iPhone — it starts broadcasting automatically.
  5. EpocCam appears as a virtual webcam device:
         SOURCE=1   (or 2 if built-in is index 0)
  6. Run:  python main.py

── RTSP IP CAMERA ───────────────────────────────────────────────────────
  URL format:  rtsp://username:password@192.168.1.x:554/stream
  Example:     SOURCE=rtsp://admin:admin@192.168.1.100:554/h264Preview_01_main
  Run:         python main.py

"""


if __name__ == "__main__":
    print(get_mobile_camera_instructions())
    reader = StreamReader(0)
    if reader.connect():
        frame = reader.read_frame()
        if frame is not None:
            print(f"[test] Frame shape: {frame.shape}")
        reader.release()
