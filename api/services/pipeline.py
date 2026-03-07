"""
Integrated Pipeline Service
============================
Refactored from src/detection/integrated_pipeline.py into a reusable,
API-friendly service.  No cv2.imshow / GUI dependencies — returns pure
data structures that FastAPI can serialise to JSON.
"""

import cv2
import math
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Optional


# ──────────────────────────────────────────────
# Centroid Tracker  (class-aware, with history)
# ──────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_history: int = 30):
        self.next_object_id = 0
        self.objects: dict[int, tuple] = {}
        self.disappeared: dict[int, int] = {}
        self.obj_labels: dict[int, str] = {}
        self.tracks: dict[int, list] = defaultdict(list)
        self.max_disappeared = max_disappeared
        self.max_history = max_history

    def register(self, centroid, label=None):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.obj_labels[self.next_object_id] = label or "unknown"
        self.tracks[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        for store in (self.objects, self.disappeared, self.obj_labels, self.tracks):
            store.pop(object_id, None)

    def update(self, detections):
        """
        detections: list of (centroid,) or ((cx,cy), label) tuples
        Returns dict  object_id -> centroid
        """
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids, input_labels = [], []
        for d in detections:
            if isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], tuple):
                input_centroids.append(d[0])
                input_labels.append(d[1])
            else:
                input_centroids.append(d)
                input_labels.append(None)

        if not self.objects:
            for i, c in enumerate(input_centroids):
                self.register(c, input_labels[i])
            return self.objects

        oids = list(self.objects.keys())
        ocentroids = list(self.objects.values())

        D = np.zeros((len(ocentroids), len(input_centroids)))
        for i, oc in enumerate(ocentroids):
            for j, ic in enumerate(input_centroids):
                D[i][j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            existing_label = self.obj_labels.get(oids[row])
            new_label = input_labels[col]
            threshold = 150 if "ball" in (existing_label, new_label) else 50

            if D[row, col] > threshold:
                continue

            oid = oids[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            if new_label is not None:
                self.obj_labels[oid] = new_label
            self.tracks[oid].append(input_centroids[col])
            if len(self.tracks[oid]) > self.max_history:
                self.tracks[oid] = self.tracks[oid][-self.max_history:]
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])) - used_rows:
            oid = oids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in set(range(D.shape[1])) - used_cols:
            self.register(input_centroids[col], input_labels[col])

        return self.objects

    def get_label(self, oid):
        return self.obj_labels.get(oid, "unknown")

    def get_track(self, oid):
        return list(self.tracks.get(oid, []))


# ──────────────────────────────────────────────
# Team Classifier
# ──────────────────────────────────────────────
BALL_LABELS = {"ball", "sports ball", "soccer ball", "football"}
REFEREE_LABELS = {"referee", "official", "ref"}
PLAYER_LABELS = {"player", "person"}
TEAM_SAMPLE_MIN = 60
TEAM_COLOR_MATCH_THRESHOLD = 45


def _bgr_to_lab(color_bgr):
    c = np.uint8([[color_bgr]])
    return cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def _lab_distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _get_dominant_color(image):
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return np.array([0, 0, 0])
    upper = image[: int(image.shape[0] * 0.6), :]
    if upper.size == 0:
        return np.array([0, 0, 0])
    pixels = upper.reshape((-1, 3))
    mask = ~((pixels < 50).all(axis=1) | (pixels > 200).all(axis=1))
    filtered = pixels[mask]
    if len(filtered) < 10:
        filtered = pixels
    km = KMeans(n_clusters=1, n_init=10, random_state=42)
    km.fit(filtered)
    return km.cluster_centers_[0]


class TeamClassifier:
    """Stateful team classifier that seeds itself from jersey colors."""

    def __init__(self):
        self.team_colors: list[np.ndarray] = []
        self.team_colors_lab: list[np.ndarray] = []
        self.samples: list[np.ndarray] = []
        self.initialized = False
        self.player_team_map: dict[int, str] = {}

    def classify(self, player_id: int, color: np.ndarray) -> str:
        if player_id in self.player_team_map:
            return self.player_team_map[player_id]

        color = np.array(color, dtype=float)
        color_lab = _bgr_to_lab(color)

        if self.initialized and len(self.team_colors_lab) >= 2:
            d0 = _lab_distance(color_lab, self.team_colors_lab[0])
            d1 = _lab_distance(color_lab, self.team_colors_lab[1])
            team = "Other" if min(d0, d1) > TEAM_COLOR_MATCH_THRESHOLD else ("Team A" if d0 < d1 else "Team B")
            self.player_team_map[player_id] = team
            return team

        self.samples.append(color)
        if len(self.samples) >= TEAM_SAMPLE_MIN and not self.initialized:
            try:
                km = KMeans(n_clusters=2, n_init=10, random_state=42)
                km.fit(np.array(self.samples))
                self.team_colors = [km.cluster_centers_[0], km.cluster_centers_[1]]
                self.team_colors_lab = [_bgr_to_lab(c) for c in self.team_colors]
                self.initialized = True
            except Exception:
                pass

        if len(self.team_colors) == 0:
            team = "Team A"
            self.team_colors.append(color)
            self.team_colors_lab.append(_bgr_to_lab(color))
        elif len(self.team_colors) == 1:
            if np.linalg.norm(color - self.team_colors[0]) > TEAM_COLOR_MATCH_THRESHOLD * 1.2:
                team = "Team B"
                self.team_colors.append(color)
                self.team_colors_lab.append(_bgr_to_lab(color))
            else:
                team = "Team A"
        else:
            d0 = np.linalg.norm(color - self.team_colors[0])
            d1 = np.linalg.norm(color - self.team_colors[1])
            team = "Team A" if d0 < d1 else "Team B"

        self.player_team_map[player_id] = team
        return team


# ──────────────────────────────────────────────
# Integrated Pipeline (stateful, per-run)
# ──────────────────────────────────────────────
class IntegratedPipeline:
    """
    Run the full detection → tracking → team classification → possession
    pipeline on a video file.  All state is instance-scoped so multiple
    videos can be analysed concurrently.
    """

    def __init__(self, model_path: str = "models/best.pt"):
        self.model = YOLO(model_path)
        self.tracker = CentroidTracker(max_disappeared=30)
        self.classifier = TeamClassifier()

    # ---- helpers ----
    @staticmethod
    def _center(x1, y1, x2, y2):
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ---- main entry ----
    def run(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> dict:
        """
        Process the video and return structured analytics.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.
        max_frames : int, optional
            Stop after this many frames (useful for quick demos).
        progress_callback : callable, optional
            Called with (frames_processed, total_frames) periodically.

        Returns
        -------
        dict   with keys:
            frames_processed, total_video_frames, fps,
            possession, player_tracking, ball_tracking
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"INFO: Video opened. FPS={fps}, Total Frames={total_video_frames}")

        possession_counter = {"Team A": 0, "Team B": 0}
        player_positions: dict[int, list[dict]] = defaultdict(list)   # id -> [{frame, x, y, team}]
        player_distances: dict[int, float] = defaultdict(float)
        ball_positions: list[dict] = []                                # [{frame, x, y}]

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_idx == 0:
                    print("ERROR: Failed to read the very first frame!")
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            results = self.model(frame)[0]

            centroids = []
            det_data = []
            ball_pos = None

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_label = self.model.names[cls].lower()

                if raw_label in PLAYER_LABELS:
                    obj_cls = "player"
                elif raw_label in BALL_LABELS:
                    obj_cls = "ball"
                elif raw_label in REFEREE_LABELS:
                    obj_cls = "referee"
                else:
                    obj_cls = raw_label

                if obj_cls in {"player", "ball", "referee"}:
                    cx, cy = self._center(x1, y1, x2, y2)
                    centroids.append(((cx, cy), obj_cls))
                    det_data.append((x1, y1, x2, y2, cx, cy, obj_cls, conf))

            tracked = self.tracker.update(centroids)

            # ---- match detections to tracked IDs ----
            for x1, y1, x2, y2, cx, cy, obj_cls, conf in det_data:
                oid = None
                for _id, cent in tracked.items():
                    if abs(cent[0] - cx) < 10 and abs(cent[1] - cy) < 10:
                        oid = _id
                        break
                if oid is None:
                    continue

                if obj_cls == "player":
                    h1, h2 = max(0, y1), min(frame.shape[0], y2)
                    w1, w2 = max(0, x1), min(frame.shape[1], x2)
                    crop = frame[h1:h2, w1:w2]
                    if oid not in self.classifier.player_team_map:
                        color = _get_dominant_color(crop)
                        self.classifier.classify(oid, color)
                    team = self.classifier.player_team_map.get(oid, "Unknown")

                    # record position
                    prev = player_positions[oid][-1] if player_positions[oid] else None
                    player_positions[oid].append({
                        "frame": frame_idx,
                        "x": cx,
                        "y": cy,
                        "team": team,
                        "confidence": round(conf, 3),
                    })
                    if prev:
                        player_distances[oid] += self._distance(
                            (prev["x"], prev["y"]), (cx, cy)
                        )

                elif obj_cls == "ball":
                    ball_pos = (cx, cy)
                    ball_positions.append({"frame": frame_idx, "x": cx, "y": cy})

            # ---- possession ----
            if ball_pos:
                closest_team = None
                closest_dist = float("inf")
                for _id, cent in tracked.items():
                    if self.tracker.get_label(_id) != "player":
                        continue
                    t = self.classifier.player_team_map.get(_id)
                    if t and t in ("Team A", "Team B"):
                        d = self._distance(cent, ball_pos)
                        if d < closest_dist:
                            closest_dist = d
                            closest_team = t
                if closest_team:
                    possession_counter[closest_team] += 1

            # progress
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_video_frames)

        cap.release()

        # ---- compile results ----
        total_pos = possession_counter["Team A"] + possession_counter["Team B"]
        if total_pos > 0:
            poss_a = round(100 * possession_counter["Team A"] / total_pos, 2)
            poss_b = round(100 * possession_counter["Team B"] / total_pos, 2)
        else:
            poss_a = poss_b = 0.0

        # serialise player tracking (convert defaultdict → plain dict, id → str key)
        tracking_data = {}
        for pid, positions in player_positions.items():
            team = positions[0]["team"] if positions else "Unknown"
            tracking_data[str(pid)] = {
                "team": team,
                "total_distance_px": round(player_distances.get(pid, 0), 2),
                "positions": positions,
            }

        return {
            "frames_processed": frame_idx,
            "total_video_frames": total_video_frames,
            "fps": fps,
            "possession": {
                "team_a": poss_a,
                "team_b": poss_b,
                "team_a_frames": possession_counter["Team A"],
                "team_b_frames": possession_counter["Team B"],
            },
            "players_tracked": len(tracking_data),
            "player_tracking": tracking_data,
            "ball_tracking": ball_positions,
        }
