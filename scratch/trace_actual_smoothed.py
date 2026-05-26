#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["shoplifting/shoplifting-10.mp4"]
detections = video["detections"]

# Cluster
raw_shelf_detections = []
for frame_num, detections_frame in enumerate(detections):
    persons = [d for d in detections_frame if d['type'] == 'person']
    objects = [d for d in detections_frame if d['type'] not in ['backpack', 'handbag', 'person']]
    
    for obj in objects:
        overlaps = False
        for person in persons:
            x1_1, y1_1, x2_1, y2_1 = person['bbox']
            x1_2, y1_2, x2_2, y2_2 = obj['bbox']
            if not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2):
                overlaps = True
                break
        if not overlaps:
            ox1, oy1, ox2, oy2 = obj['bbox']
            cx = (ox1 + ox2) / 2
            cy = (oy1 + oy2) / 2
            raw_shelf_detections.append((cx, cy))
            
slots = []
for cx, cy in raw_shelf_detections:
    found = False
    for slot in slots:
        scx, scy = slot['center']
        if ((cx - scx)**2 + (cy - scy)**2)**0.5 < 30.0:
            n = slot['count']
            slot['center'] = ((scx * n + cx) / (n + 1), (scy * n + cy) / (n + 1))
            slot['count'] += 1
            found = True
            break
    if not found:
        slots.append({'center': (cx, cy), 'count': 1})
        
shelf_slots = [s for s in slots if s['count'] >= 15]

# Calculate counts
slot_occupancy_counts = []
for frame_idx, detections_frame in enumerate(detections):
    objects = [d for d in detections_frame if d['type'] not in ['backpack', 'handbag', 'person']]
    frame_counts = []
    for slot_idx, slot in enumerate(shelf_slots):
        scx, scy = slot['center']
        count = 0
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            dist = ((ocx - scx) ** 2 + (ocy - scy) ** 2) ** 0.5
            if dist < 45.0:
                count += 1
        frame_counts.append(count)
    slot_occupancy_counts.append(frame_counts)

# Smooth counts
smoothed_occupancy_counts = []
for frame_idx in range(len(slot_occupancy_counts)):
    frame_smoothed = []
    for slot_idx in range(len(shelf_slots)):
        window = []
        for f in range(max(0, frame_idx - 2), min(len(slot_occupancy_counts), frame_idx + 3)):
            window.append(slot_occupancy_counts[f][slot_idx])
        frame_smoothed.append(sorted(window)[len(window)//2])
    smoothed_occupancy_counts.append(frame_smoothed)

for slot_idx, slot in enumerate(shelf_slots):
    print(f"Slot {slot_idx} center={slot['center']}, raw max={max([c[slot_idx] for c in slot_occupancy_counts])}, smoothed max={max([s[slot_idx] for s in smoothed_occupancy_counts])}")
