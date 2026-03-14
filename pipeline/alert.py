"""
VisionGuard AI — Alert Engine
Handles sound playback, file saving, database logging, and console output.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


# ── ANSI colour helpers ──────────────────────────────────────────────────────

_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ── SQLite DB ────────────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            camera_id     TEXT    NOT NULL,
            track_pair    TEXT,
            confidence    REAL    NOT NULL,
            snapshot_path TEXT,
            clip_path     TEXT,
            false_alarm   INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


# ── Alert sound generation ───────────────────────────────────────────────────

def _generate_alert_wav(path: Path) -> None:
    """Programmatically generate a 880 Hz beep and save as WAV."""
    sample_rate = 44100
    duration = 0.5
    freq = 880.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_data = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())
    print(f"[Alert] Generated alert sound: {path}")


def _play_sound_async(sound_path: Path) -> None:
    """Play alert.wav in a daemon thread (non-blocking)."""
    def _play():
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(sound_path))
            pygame.mixer.music.play()
            # Wait for playback to finish before re-initialising
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as exc:
            print(f"[Alert] Sound playback failed: {exc}")

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ── Main Alert Engine ────────────────────────────────────────────────────────

class AlertEngine:
    """
    Manages the full alert lifecycle:
    - Persistence counter (N consecutive positives → alert)
    - Per-pair cooldown timer
    - Sound, snapshot, clip, DB log, console output
    """

    def __init__(
        self,
        cooldown_seconds: int = 60,
        persistence_count: int = 2,
        save_clips: bool = True,
        save_snapshots: bool = True,
        db_path: Path = config.DB_PATH,
        alerts_dir: Path = config.ALERTS_DIR,
        sound_path: Path = config.ALERT_SOUND,
    ):
        self.cooldown = cooldown_seconds
        self.persistence_count = persistence_count
        self.save_clips = save_clips
        self.save_snapshots = save_snapshots
        self.alerts_dir = Path(alerts_dir)
        self.sound_path = Path(sound_path)

        # Ensure sound file exists
        if not self.sound_path.exists():
            _generate_alert_wav(self.sound_path)

        # DB
        self._conn = _init_db(Path(db_path))
        self._db_lock = threading.Lock()

        # Per-(camera_id, pair) state
        self._consecutive: Dict[tuple, int] = {}
        self._last_alert_time: Dict[tuple, float] = {}

        # Pending clip frames for in-progress recordings
        self._clip_frames: Dict[tuple, List[np.ndarray]] = {}

        print("[Alert] AlertEngine ready.")

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        camera_id: str,
        track_pair: Tuple[int, int],
        confidence: float,
        frame: np.ndarray,
        clip_frames: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """
        Feed a classifier result into the alert pipeline.

        Parameters
        ----------
        camera_id   : identifier string for the camera source
        track_pair  : (track_id_1, track_id_2)
        confidence  : P(violent) from the classifier
        frame       : current BGR frame (for snapshot)
        clip_frames : list of BGR frames forming the raw clip (for MP4 save)

        Returns
        -------
        bool — True if an alert was fired this call.
        """
        key = (camera_id, min(track_pair), max(track_pair))
        now = time.time()

        # Accumulate raw frames for potential clip save
        if clip_frames:
            self._clip_frames[key] = clip_frames

        from pipeline.classifier import ClipClassifier  # lazy import
        is_violent = confidence >= config.CLASSIFIER_THRESHOLD

        if is_violent:
            self._consecutive[key] = self._consecutive.get(key, 0) + 1
        else:
            self._consecutive[key] = 0
            return False

        if self._consecutive[key] < self.persistence_count:
            return False

        # Check cooldown
        last = self._last_alert_time.get(key, 0.0)
        if now - last < self.cooldown:
            return False

        # ── FIRE ALERT ─────────────────────────────────────────────────────
        self._consecutive[key] = 0
        self._last_alert_time[key] = now

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path: Optional[Path] = None
        clip_path: Optional[Path] = None

        if self.save_snapshots:
            snap_path = self.alerts_dir / f"{ts}_{camera_id}_snap.jpg"
            cv2.imwrite(str(snap_path), frame)

        if self.save_clips and key in self._clip_frames:
            clip_path = self.alerts_dir / f"{ts}_{camera_id}_clip.mp4"
            self._save_clip(self._clip_frames[key], clip_path)

        incident_id = self._log_to_db(
            timestamp=ts,
            camera_id=camera_id,
            track_pair=str(track_pair),
            confidence=confidence,
            snapshot_path=str(snap_path) if snap_path else None,
            clip_path=str(clip_path) if clip_path else None,
        )

        _play_sound_async(self.sound_path)
        self._print_alert(incident_id, camera_id, track_pair, confidence, ts)

        return True

    def log_false_alarm(self, incident_id: int) -> bool:
        """Mark a DB record as a false alarm. Returns True on success."""
        with self._db_lock:
            cursor = self._conn.execute(
                "UPDATE incidents SET false_alarm=1 WHERE id=?", (incident_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def get_recent_incidents(self, n: int = 10) -> List[dict]:
        """Return the n most recent incident records as dicts."""
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, camera_id, track_pair, confidence,
                       snapshot_path, clip_path, false_alarm
                FROM incidents
                ORDER BY id DESC LIMIT ?
                """,
                (n,),
            ).fetchall()
        keys = [
            "id", "timestamp", "camera_id", "track_pair", "confidence",
            "snapshot_path", "clip_path", "false_alarm",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def get_incident(self, incident_id: int) -> Optional[dict]:
        """Fetch a single incident by ID."""
        with self._db_lock:
            row = self._conn.execute(
                """
                SELECT id, timestamp, camera_id, track_pair, confidence,
                       snapshot_path, clip_path, false_alarm
                FROM incidents WHERE id=?
                """,
                (incident_id,),
            ).fetchone()
        if row is None:
            return None
        keys = [
            "id", "timestamp", "camera_id", "track_pair", "confidence",
            "snapshot_path", "clip_path", "false_alarm",
        ]
        return dict(zip(keys, row))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_clip(self, frames: List[np.ndarray], path: Path) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    def _log_to_db(
        self,
        timestamp: str,
        camera_id: str,
        track_pair: str,
        confidence: float,
        snapshot_path: Optional[str],
        clip_path: Optional[str],
    ) -> int:
        with self._db_lock:
            cursor = self._conn.execute(
                """
                INSERT INTO incidents
                (timestamp, camera_id, track_pair, confidence,
                 snapshot_path, clip_path, false_alarm)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (timestamp, camera_id, track_pair, confidence,
                 snapshot_path, clip_path),
            )
            self._conn.commit()
        return cursor.lastrowid

    @staticmethod
    def _print_alert(
        incident_id: int,
        camera_id: str,
        track_pair: tuple,
        confidence: float,
        timestamp: str,
    ) -> None:
        bar_len = int(confidence * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"\n{_RED}{_BOLD}"
            f"{'═' * 60}\n"
            f"  ⚠  VIOLENCE DETECTED  —  Incident #{incident_id}\n"
            f"{'═' * 60}{_RESET}\n"
            f"  Camera   : {camera_id}\n"
            f"  Pair     : tracks {track_pair[0]} & {track_pair[1]}\n"
            f"  Conf     : {_RED}{bar}{_RESET} {confidence:.1%}\n"
            f"  Time     : {timestamp}\n"
        )
