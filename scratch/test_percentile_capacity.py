#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["normal/normal-11.mp4"]
detections = video["detections"]

shelf_slots = [
    {'center': (263.2, 204.5)}  # Slot 2
]

for f in range(len(detections)):
    items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
    persons = [d for d in detections[f] if d['type'] == 'person']
    
    # Check if person is near this slot
    scx, scy = shelf_slots[0]['center']
    person_near = False
    for p in persons:
        px1, py1, px2, py2 = p['bbox']
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        if ((pcx - scx)**2 + (pcy - scy)**2)**0.5 < 120.0:
            person_near = True
            break
            
    # Calculate count
    count = 0
    for item in items:
        x1, y1, x2, y2 = item['bbox']
        icx = (x1 + x2) / 2
        icy = (y1 + y2) / 2
        if ((icx - scx)**2 + (icy - scy)**2)**0.5 < 45.0:
            count += 1
            
    if not person_near:
        print(f"Frame {f} (CLEAN): count={count}")
