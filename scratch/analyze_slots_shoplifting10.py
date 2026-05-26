#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["shoplifting/shoplifting-10.mp4"]
detections = video["detections"]

# Shelf slots from diagnose output
shelf_slots = [
    (167.1, 294.3), # Slot 0
    (48.2, 316.0),  # Slot 1
    (89.6, 264.4),  # Slot 2
    (447.3, 466.5), # Slot 3
    (12.5, 227.4)   # Slot 4
]

for slot_idx, slot in enumerate(shelf_slots):
    counts = []
    for f in range(len(detections)):
        items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
        count = 0
        for item in items:
            x1, y1, x2, y2 = item['bbox']
            icx = (x1 + x2) / 2
            icy = (y1 + y2) / 2
            if ((icx - slot[0])**2 + (icy - slot[1])**2)**0.5 < 45.0:
                count += 1
        counts.append(count)
    
    start_avg = sum(counts[:30]) / 30
    end_avg = sum(counts[-30:]) / 30
    print(f"Slot {slot_idx}: start={start_avg:.2f}, end={end_avg:.2f}, diff={end_avg - start_avg:.2f}, max={max(counts)}")
