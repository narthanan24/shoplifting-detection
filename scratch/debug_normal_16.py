import pickle
from fast_eval import SimulatedDetector

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)

video_data = cache['normal/normal-16.mp4']
frames = video_data['detections']

detector = SimulatedDetector()
# We will intercept process_cached_video
metadata = video_data['metadata']
frames_detections = video_data['detections']
fps = metadata['fps']
duration = metadata['duration']

# Identify static slots
raw_shelf_detections = []
for frame_num, detections in enumerate(frames_detections):
    persons = [d for d in detections if d['type'] == 'person']
    objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
    
    for obj in objects:
        overlaps_person = False
        for person in persons:
            if detector._check_overlap(person['bbox'], obj['bbox']):
                overlaps_person = True
                break
        if not overlaps_person:
            ox1, oy1, ox2, oy2 = obj['bbox']
            cx = (ox1 + ox2) / 2
            cy = (oy1 + oy2) / 2
            raw_shelf_detections.append((cx, cy))

slots = []
for cx, cy in raw_shelf_detections:
    found = False
    for slot in slots:
        scx, scy = slot['center']
        dist = ((cx - scx) ** 2 + (cy - scy) ** 2) ** 0.5
        if dist < 30.0:
            n = slot['count']
            slot['center'] = ((scx * n + cx) / (n + 1), (scy * n + cy) / (n + 1))
            slot['count'] += 1
            found = True
            break
    if not found:
        slots.append({'center': (cx, cy), 'count': 1})

shelf_slots = [s for s in slots if s['count'] >= 15]
print("Shelf slots:")
for i, s in enumerate(shelf_slots):
    print(f"  Slot {i}: center={s['center']}, count={s['count']}")

# Run the frames and log potential pickups details
from tracker import ByteTracker
from collections import defaultdict

tracker = ByteTracker()
potential_pickups = defaultdict(list)
slot_occupancy = []

person_last_seen_frame = {}
person_first_seen_frame = {}

for frame_num, detections in enumerate(frames_detections):
    persons = [d for d in detections if d['type'] == 'person']
    objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
    bags = [d for d in detections if d['type'] in ['backpack', 'handbag']]
    
    tracked_persons = tracker.update(detections)
    
    current_occupancy = []
    for slot_idx, slot in enumerate(shelf_slots):
        scx, scy = slot['center']
        occupied = False
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            ocx = (ox1 + ocx) / 2 if 'ocx' in locals() else (ox1 + ox2) / 2
            # Let's fix ocx/ocy calculation
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            dist = ((ocx - scx) ** 2 + (ocy - scy) ** 2) ** 0.5
            if dist < 40.0:
                occupied = True
                break
        current_occupancy.append(occupied)
    slot_occupancy.append(current_occupancy)
    
    for person in tracked_persons:
        track_id = person['track_id']
        px1, py1, px2, py2 = person['bbox']
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        
        if track_id not in person_first_seen_frame:
            person_first_seen_frame[track_id] = frame_num
        person_last_seen_frame[track_id] = frame_num
        
        for slot_idx, slot in enumerate(shelf_slots):
            scx, scy = slot['center']
            is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
            dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
            
            if is_inside or dist < 130.0:
                if not current_occupancy[slot_idx]:
                    existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                    if not existing:
                        has_bag = False
                        for bag in bags:
                            if detector._check_overlap(person['bbox'], bag['bbox']):
                                has_bag = True
                                break
                        print(f"Frame {frame_num} ({frame_num/fps:.2f}s): Person {track_id} potential pickup Slot {slot_idx} (dist={dist:.1f}, has_bag={has_bag})")
                        potential_pickups[track_id].append({
                            'slot_idx': slot_idx,
                            'frame_pickup': frame_num,
                            'max_dist': dist,
                            'cancelled': False,
                            'has_bag': has_bag,
                            'confirmed': False
                        })
        
        for pickup in potential_pickups[track_id]:
            if pickup['cancelled'] or pickup['confirmed']:
                continue
            slot_idx = pickup['slot_idx']
            scx, scy = shelf_slots[slot_idx]['center']
            dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
            pickup['max_dist'] = max(pickup['max_dist'], dist)
            
            if dist > 160.0:
                was_occupied_since = False
                for f in range(pickup['frame_pickup'], frame_num + 1):
                    if slot_occupancy[f][slot_idx]:
                        was_occupied_since = True
                        break
                if was_occupied_since:
                    print(f"Frame {frame_num}: Pickup for Person {track_id} Slot {slot_idx} CANCELLED (occupied again)")
                    pickup['cancelled'] = True

# Post check
for track_id, pickups in potential_pickups.items():
    for pickup in pickups:
        slot_idx = pickup['slot_idx']
        was_occupied_before = any(slot_occupancy[f][slot_idx] for f in range(0, pickup['frame_pickup']))
        was_occupied_after = any(slot_occupancy[f][slot_idx] for f in range(pickup['frame_pickup'], len(frames_detections)))
        print(f"\nPickup summary for Person {track_id} Slot {slot_idx}:")
        print(f"  Cancelled: {pickup['cancelled']}")
        print(f"  Was occupied before pickup frame: {was_occupied_before}")
        print(f"  Was occupied after pickup frame: {was_occupied_after}")
