from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

def update_tracker(detections):
    tracks = tracker.update_tracks(detections)
    return tracks
