#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["normal/normal-11.mp4"]
detections = video["detections"]

# Calculate counts for Slot 2
counts_slot2 = []
for f in range(len(detections)):
    items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
    slot_2_cnt = 0
    for item in items:
        x1, y1, x2, y2 = item['bbox']
        icx = (x1 + x2) / 2
        icy = (y1 + y2) / 2
        dist2 = ((icx - 263.2)**2 + (icy - 204.5)**2)**0.5
        if dist2 < 45.0:
            slot_2_cnt += 1
    counts_slot2.append(slot_2_cnt)

# Test different sizes
for size in [5, 9, 15, 25]:
    # Smooth
    window_half = size // 2
    smoothed = []
    for frame_idx in range(len(counts_slot2)):
        window = []
        for f in range(max(0, frame_idx - window_half), min(len(counts_slot2), frame_idx + window_half + 1)):
            window.append(counts_slot2[f])
        smoothed.append(sorted(window)[len(window)//2])
    
    print(f"Size {size} max: {max(smoothed)}")
