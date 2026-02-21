import cv2
import math
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load once
model = YOLO("models/best.pt")
tracker = DeepSort(max_age=30)


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    possession = {"Team 1": 0, "Team 2": 0}
    player_positions = {}
    player_distance = {}

    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model(frame)[0]

        detections = []
        ball_position = None
        tracked_players = []

        # -----------------------------
        # YOLO detections
        # -----------------------------
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "player":
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])
                detections.append(([x1, y1, w, h], conf, label))

            if label == "ball":
                cx = int((x1 + x2)/2)
                cy = int((y1 + y2)/2)
                ball_position = (cx, cy)

        # -----------------------------
        # Tracking
        # -----------------------------
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            # Dummy team assignment (simple version)
            team = "Team 1" if tid % 2 == 0 else "Team 2"

            tracked_players.append((tid, team, (cx, cy)))

            # distance tracking
            if tid in player_positions:
                d = distance(player_positions[tid], (cx, cy))
                player_distance[tid] = player_distance.get(tid, 0) + d

            player_positions[tid] = (cx, cy)

        # -----------------------------
        # Possession logic
        # -----------------------------
        if ball_position and tracked_players:
            closest = min(
                tracked_players,
                key=lambda p: distance(p[2], ball_position)
            )
            possession[closest[1]] += 1

        # Optional speed limit (demo)
        if total_frames > 300:
            break

    cap.release()

    # Final stats
    total_pos = possession["Team 1"] + possession["Team 2"]

    if total_pos > 0:
        p1 = 100 * possession["Team 1"] / total_pos
        p2 = 100 * possession["Team 2"] / total_pos
    else:
        p1 = p2 = 0

    return {
        "frames_processed": total_frames,
        "possession": {
            "team1": round(p1, 2),
            "team2": round(p2, 2)
        },
        "players_tracked": len(player_distance),
        "player_distances": player_distance
    }
