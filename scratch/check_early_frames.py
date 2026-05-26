#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["normal/normal-11.mp4"]
detections = video["detections"]

# Slot centers
# Slot 0: center=(254.3, 239.1)
# Slot 1: center=(420.1, 362.7)
# Slot 2: center=(263.2, 204.5)

for f in range(25, 60):
    items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
    slot_0_cnt = 0
    slot_2_cnt = 0
    for item in items:
        x1, y1, x2, y2 = item['bbox']
        icx = (x1 + x2) / 2
        icy = (y1 + y2) / 2
        dist0 = ((icx - 254.3)**2 + (icy - 239.1)**2)**0.5
        dist2 = ((icx - 263.2)**2 + (icy - 204.5)**2)**0.5
        if dist0 < 45.0:
            slot_0_cnt += 1
        if dist2 < 45.0:
            slot_2_cnt += 1
    print(f"Frame {f}: Slot 0 count={slot_0_cnt}, Slot 2 count={slot_2_cnt}")
