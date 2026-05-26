#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)
video = cache["shoplifting/shoplifting-3.mp4"]
detections = video["detections"]

# Cluster raw shelf detections
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
print(f"Detected {len(shelf_slots)} slots")

for slot_idx, slot in enumerate(shelf_slots):
    counts = []
    for f in range(len(detections)):
        items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
        count = 0
        for item in items:
            x1, y1, x2, y2 = item['bbox']
            icx = (x1 + x2) / 2
            icy = (y1 + y2) / 2
            if ((icx - slot['center'][0])**2 + (icy - slot['center'][1])**2)**0.5 < 45.0:
                count += 1
        counts.append(count)
    
    start_avg = sum(counts[:30]) / 30
    end_avg = sum(counts[-30:]) / 30
    print(f"Slot {slot_idx} (center={slot['center']}): start={start_avg:.2f}, end={end_avg:.2f}, diff={end_avg - start_avg:.2f}, max={max(counts)}")
