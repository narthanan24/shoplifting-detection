#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["shoplifting/shoplifting-10.mp4"]
detections = video["detections"]

# Slot 0 center: (167.1, 294.3)
tracker = ByteTracker()

for f in range(len(detections)):
    tracked_persons = tracker.update(detections[f])
    for p in tracked_persons:
        if p['track_id'] == 1:
            px1, py1, px2, py2 = p['bbox']
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            dist = ((pcx - 167.1)**2 + (pcy - 294.3)**2)**0.5
            if 140 <= f <= 160:
                print(f"Frame {f}: Person 1 center=({pcx:.1f},{pcy:.1f}), dist={dist:.1f}")
