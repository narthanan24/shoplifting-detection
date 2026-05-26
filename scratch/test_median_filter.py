#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["normal/normal-11.mp4"]
detections = video["detections"]

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
    if slot_2_cnt > 1:
        print(f"Frame {f}: count={slot_2_cnt}")
