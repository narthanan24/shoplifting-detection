#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["shoplifting/shoplifting-10.mp4"]
detections = video["detections"]

# Slot 0 center: (167.1, 294.3)
counts = []
for f in range(len(detections)):
    items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
    count = 0
    for item in items:
        x1, y1, x2, y2 = item['bbox']
        icx = (x1 + x2) / 2
        icy = (y1 + y2) / 2
        if ((icx - 167.1)**2 + (icy - 294.3)**2)**0.5 < 45.0:
            count += 1
    counts.append(count)

smoothed_counts = []
for frame_idx in range(len(counts)):
    window = []
    for f in range(max(0, frame_idx - 2), min(len(counts), frame_idx + 3)):
        window.append(counts[f])
    smoothed_counts.append(sorted(window)[len(window)//2])

for f in range(0, 60):
    print(f"Frame {f}: raw={counts[f]}, smoothed={smoothed_counts[f]}")
