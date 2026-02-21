import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
import math
import time
prev_time = time.time()


player_team_map = {}

# ==============================
# Class-aware Centroid Tracker (stores class + centroid history)
# ==============================
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_history=30):
        self.next_object_id = 0
        self.objects = {}                # object_id -> (x, y)
        self.disappeared = {}            # object_id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.obj_labels = {}             # object_id -> class label (player/ball/referee/...)
        self.tracks = defaultdict(list)  # object_id -> list of centroids (history)
        self.max_history = max_history

    def register(self, centroid, label=None):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.obj_labels[self.next_object_id] = label if label is not None else "unknown"
        self.tracks[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.obj_labels:
            del self.obj_labels[object_id]
        if object_id in self.tracks:
            del self.tracks[object_id]

    def update(self, detections):
        """
        detections: list of either
          - (x,y)  OR
          - ((x,y), label)
        Returns: dict of object_id -> centroid
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # normalize input into parallel lists
        input_centroids = []
        input_labels = []
        for d in detections:
            if isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], tuple):
                input_centroids.append(d[0])
                input_labels.append(d[1])
            else:
                input_centroids.append(d)
                input_labels.append(None)

        # register if no existing objects
        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, input_labels[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # distance matrix
        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i][j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            existing_label = self.obj_labels.get(object_id)
            new_label = input_labels[col]

            # allow larger movement for small/fast objects (ball)
            threshold = 50
            if existing_label == "ball" or new_label == "ball":
                threshold = 150

            if D[row, col] > threshold:
                continue

            # update matched object
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            if new_label is not None:
                self.obj_labels[object_id] = new_label
            self.tracks[object_id].append(input_centroids[col])
            if len(self.tracks[object_id]) > self.max_history:
                self.tracks[object_id] = self.tracks[object_id][-self.max_history:]

            used_rows.add(row)
            used_cols.add(col)

        # process disappeared & newly seen
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col], input_labels[col])

        return self.objects

    def get_label(self, object_id):
        return self.obj_labels.get(object_id, "unknown")

    def get_track(self, object_id):
        return list(self.tracks.get(object_id, []))

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("models/best.pt")
tracker = CentroidTracker(max_disappeared=30)

# ==============================
# TEAM COLOR STORAGE & SEEDING
# ==============================
team_colors = []            # will hold two cluster centers (BGR)
team_colors_lab = []        # same centers in LAB (for perceptual distance)
team_color_samples = []     # collected jersey colors (for automatic seeding)
TEAM_SAMPLE_MIN = 60        # samples required before running KMeans to seed teams
team_initialized = False    # becomes True after automatic seeding completes

# color-match threshold (LAB distance). Lower -> stricter matching -> fewer false positives
TEAM_COLOR_MATCH_THRESHOLD = 45

# Mapping player id -> team name shown on-screen
# (we keep `player_team_map` at top as the single source of truth)

# Normalized labels (some models use different names)
BALL_LABELS = {"ball", "sports ball", "soccer ball", "football"}
REFEREE_LABELS = {"referee", "official", "ref"}
PLAYER_LABELS = {"player", "person"}

# How many centroids to keep for trajectory drawing
MAX_TRACK_HISTORY = 30

def get_dominant_color(image):
    """Extract dominant jersey color with better preprocessing"""
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return np.array([0, 0, 0])
    
    # Focus on upper body (jersey area)
    height = image.shape[0]
    upper_body = image[:int(height*0.6), :]
    
    if upper_body.size == 0:
        return np.array([0, 0, 0])
    
    # Reshape pixels
    pixels = upper_body.reshape((-1, 3))
    
    # Filter out dark/white colors (field lines, shorts)
    mask = ~((pixels < 50).all(axis=1) | (pixels > 200).all(axis=1))
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < 10:
        filtered_pixels = pixels
    
    # Use KMeans to find dominant color
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)
    return kmeans.cluster_centers_[0]


# --- Color distance helpers (use LAB for perceptual distance) ---
def bgr_to_lab(color_bgr):
    """Convert single BGR color (array-like) to LAB (float) using OpenCV."""
    c = np.uint8([[color_bgr]])
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return lab


def lab_distance(a, b):
    """Euclidean distance in LAB space."""
    return float(np.linalg.norm(np.array(a).astype(float) - np.array(b).astype(float)))

def classify_team(player_id, color):
    """Classify a player into Team A / Team B using jersey color (LAB distance).

    Uses `TEAM_COLOR_MATCH_THRESHOLD` to avoid forcing assignments when the
    jersey color is far from both seeded centers (will label as "Other").
    """
    global team_colors, team_colors_lab, team_color_samples, team_initialized, player_team_map, TEAM_COLOR_MATCH_THRESHOLD

    # prefer existing assignment
    if player_id in player_team_map:
        return player_team_map[player_id]

    # Safety: ensure color is numpy array (BGR)
    color = np.array(color).astype(float)
    color_lab = bgr_to_lab(color)

    # If teams already seeded, assign to nearest center (LAB distance)
    if team_initialized and len(team_colors_lab) >= 2:
        d0 = lab_distance(color_lab, team_colors_lab[0])
        d1 = lab_distance(color_lab, team_colors_lab[1])
        if min(d0, d1) > TEAM_COLOR_MATCH_THRESHOLD:
            team = "Other"
        else:
            team = "Team A" if d0 < d1 else "Team B"
        player_team_map[player_id] = team
        return team

    # Collect samples for automatic seeding (store BGR for display/counting)
    team_color_samples.append(color)

    # If enough samples collected, run KMeans to find two team colors (on BGR)
    if len(team_color_samples) >= TEAM_SAMPLE_MIN and not team_initialized:
        try:
            X = np.array(team_color_samples)
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            team_colors.clear()
            team_colors.append(np.array(centers[0]))
            team_colors.append(np.array(centers[1]))
            # store LAB equivalents for matching
            team_colors_lab.clear()
            team_colors_lab.append(bgr_to_lab(team_colors[0]))
            team_colors_lab.append(bgr_to_lab(team_colors[1]))
            team_initialized = True
        except Exception:
            # keep sampling if clustering fails
            team_initialized = False

    # Provisional assignment while seeding — use a conservative threshold
    if len(team_colors) == 0:
        team = "Team A"
        team_colors.append(color)
        team_colors_lab.append(bgr_to_lab(color))
    elif len(team_colors) == 1:
        # if color is fairly different, create Team B, else Team A
        if np.linalg.norm(color - team_colors[0]) > (TEAM_COLOR_MATCH_THRESHOLD * 1.2):
            team = "Team B"
            team_colors.append(color)
            team_colors_lab.append(bgr_to_lab(color))
        else:
            team = "Team A"
    else:
        # fallback to nearest center (BGR fallback)
        d0 = np.linalg.norm(color - team_colors[0])
        d1 = np.linalg.norm(color - team_colors[1])
        team = "Team A" if d0 < d1 else "Team B"

    player_team_map[player_id] = team
    return team

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture("data/raw/videos/match.mp4")

# Get video properties for saving output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: Save output video
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO Detection
    results = model(frame)[0]
    
    centroids = []
    detections_data = []  # Store detection info for drawing
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_label = model.names[cls].lower()

        # normalize model label to our canonical classes
        if raw_label in PLAYER_LABELS:
            obj_cls = "player"
        elif raw_label in BALL_LABELS:
            obj_cls = "ball"
        elif raw_label in REFEREE_LABELS:
            obj_cls = "referee"
        else:
            obj_cls = raw_label

        # Only track players, ball and referees
        if obj_cls in {"player", "ball", "referee"}:
            cx, cy = get_center(x1, y1, x2, y2)
            centroids.append(((cx, cy), obj_cls))
            detections_data.append((x1, y1, x2, y2, cx, cy, obj_cls))
    
    # Update tracker (now accepts (centroid, label) pairs)
    tracked_objects = tracker.update(centroids)
    
    # Draw tracked objects (players, referees, ball)
    for detection in detections_data:
        x1, y1, x2, y2, cx, cy, obj_cls = detection
        
        # Find which tracked object this centroid belongs to
        object_id = None
        for obj_id, centroid in tracked_objects.items():
            if abs(centroid[0] - cx) < 10 and abs(centroid[1] - cy) < 10:
                object_id = obj_id
                break
        
        if object_id is None:
            continue

        # --- PLAYER ---
        if obj_cls == "player":
            # safe crop bounds
            h1, h2 = max(0, y1), min(frame.shape[0], y2)
            w1, w2 = max(0, x1), min(frame.shape[1], x2)
            player_crop = frame[h1:h2, w1:w2]

            if object_id in player_team_map:
                team = player_team_map[object_id]
            else:
                color = get_dominant_color(player_crop)
                team = classify_team(object_id, color)
                player_team_map[object_id] = team

            # fixed display colors: Team A = blue, Team B = red (BGR)
            if team == "Team A":
                box_color = (255, 0, 0)   # blue
            elif team == "Team B":
                box_color = (0, 0, 255)   # red
            else:
                box_color = (0, 255, 255) # yellow for others / unknown

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{team} | ID {object_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.circle(frame, (cx, cy), 4, box_color, -1)

        # --- REFEREE ---
        elif obj_cls == "referee":
            ref_color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ref_color, 2)
            cv2.putText(frame, f"Referee | ID {object_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ref_color, 2)
            cv2.circle(frame, (cx, cy), 4, ref_color, -1)

        # --- BALL ---
        elif obj_cls == "ball":
            ball_color = (0, 255, 255)
            cv2.circle(frame, (cx, cy), 6, ball_color, -1)
            # draw trajectory from stored track
            track = tracker.get_track(object_id)
            for i in range(1, len(track)):
                cv2.line(frame, tuple(track[i-1]), tuple(track[i]), ball_color, 2)
            cv2.putText(frame, f"Ball | ID {object_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)

        # --- fallback ---
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(frame, f"{obj_cls} | ID {object_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
# Show fixed team swatches (Team A = blue, Team B = red), seeding status and match-threshold
    x_offset = 10
    y_offset = 20
    teamA_color_display = (255, 0, 0)  # blue (BGR)
    teamB_color_display = (0, 0, 255)  # red (BGR)

    # main swatches (fixed display colors)
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 30, y_offset + 30), teamA_color_display, -1)
    cv2.putText(frame, "Team A", (x_offset + 40, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.rectangle(frame, (x_offset, y_offset + 40), (x_offset + 30, y_offset + 70), teamB_color_display, -1)
    cv2.putText(frame, "Team B", (x_offset + 40, y_offset + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # seeding progress text
    cv2.putText(frame, f"Seeding samples: {len(team_color_samples)}/{TEAM_SAMPLE_MIN}", (x_offset, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Match thr: {TEAM_COLOR_MATCH_THRESHOLD}  (- / = to adjust)", (x_offset, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    # if detected cluster centers exist, show them as small "detected" swatches
    for i, center in enumerate(team_colors[:2]):
        det_swatch = tuple(map(int, np.round(center)))
        cv2.rectangle(frame, (x_offset + 120 + i*40, y_offset), (x_offset + 140 + i*40, y_offset + 20), det_swatch, -1)
        cv2.putText(frame, f"det {i+1}", (x_offset + 120 + i*40, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.putText(frame, "Press 'r' to reset team seeding", (x_offset, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow("Football AI - Week 2 (Centroid Tracker)", frame)
    # out.write(frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        # reset team seeding and assignments
        team_colors.clear()
        team_colors_lab.clear()
        team_color_samples.clear()
        player_team_map.clear()
        team_initialized = False
    elif key == ord('='):
        TEAM_COLOR_MATCH_THRESHOLD += 5
    elif key == ord('-'):
        TEAM_COLOR_MATCH_THRESHOLD = max(5, TEAM_COLOR_MATCH_THRESHOLD - 5)

cap.release()
# out.release()
cv2.destroyAllWindows()