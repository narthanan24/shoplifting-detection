#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def process_video(video_data):
    metadata = video_data['metadata']
    frames_detections = video_data['detections']
    fps = metadata['fps']
    duration = metadata['duration']
    
    # 1. Identify static shelf slots
    raw_shelf_detections = []
    for frame_num, detections in enumerate(frames_detections):
        persons = [d for d in detections if d['type'] == 'person']
        objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
        
        for obj in objects:
            overlaps_person = False
            for person in persons:
                x1_1, y1_1, x2_1, y2_1 = person['bbox']
                x1_2, y1_2, x2_2, y2_2 = obj['bbox']
                if not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2):
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
    
    tracker = ByteTracker()
    person_movement_history = defaultdict(list)
    person_first_seen_frame = {}
    person_last_seen_frame = {}
    
    potential_pickups = defaultdict(list)
    slot_occupancy = []
    
    for frame_num, detections in enumerate(frames_detections):
        timestamp = frame_num / fps if fps > 0 else 0.0
        
        persons = [d for d in detections if d['type'] == 'person']
        objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
        bags = [d for d in detections if d['type'] in ['backpack', 'handbag']]
        
        # Track persons
        tracked_persons = tracker.update(detections)
        
        # Calculate current frame occupancy for each slot
        current_occupancy = []
        for slot_idx, slot in enumerate(shelf_slots):
            scx, scy = slot['center']
            occupied = False
            for obj in objects:
                ox1, oy1, ox2, oy2 = obj['bbox']
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                dist = ((ocx - scx) ** 2 + (ocy - scy) ** 2) ** 0.5
                if dist < 40.0:
                    occupied = True
                    break
            current_occupancy.append(occupied)
        slot_occupancy.append(current_occupancy)
        
        # Update person status
        for person in tracked_persons:
            track_id = person['track_id']
            px1, py1, px2, py2 = person['bbox']
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            
            if track_id not in person_first_seen_frame:
                person_first_seen_frame[track_id] = frame_num
            person_last_seen_frame[track_id] = frame_num
            
            # Check interaction with each slot
            for slot_idx, slot in enumerate(shelf_slots):
                scx, scy = slot['center']
                
                is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                
                if is_inside or dist < 130.0:
                    # Check if the slot is currently unoccupied but was occupied earlier
                    if not current_occupancy[slot_idx]:
                        existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                        if not existing:
                            # Check if they carry a bag
                            has_bag = False
                            for bag in bags:
                                bx1, by1, bx2, by2 = bag['bbox']
                                if not (px1 > bx2 or px2 < bx1 or py1 > by2 or py2 < by1):
                                    has_bag = True
                                    break
                            
                            potential_pickups[track_id].append({
                                'slot_idx': slot_idx,
                                'frame_pickup': frame_num,
                                'max_dist': dist,
                                'cancelled': False,
                                'has_bag': has_bag,
                                'confirmed': False
                            })
            
            # Check if person has moved away and if slot was occupied again
            for pickup in potential_pickups[track_id]:
                if pickup['cancelled'] or pickup['confirmed']:
                    continue
                    
                slot_idx = pickup['slot_idx']
                scx, scy = shelf_slots[slot_idx]['center']
                dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                pickup['max_dist'] = max(pickup['max_dist'], dist)
                
                if dist > 160.0:
                    # Check if the slot has been occupied in any frame since the pickup
                    was_occupied_since = False
                    for f in range(pickup['frame_pickup'], frame_num + 1):
                        if slot_occupancy[f][slot_idx]:
                            was_occupied_since = True
                            break
                    if was_occupied_since:
                        pickup['cancelled'] = True
    
    suspicious_events = []
    # Post-video analysis: verify and flag events
    for track_id, pickups in potential_pickups.items():
        last_seen = person_last_seen_frame[track_id]
        
        for pickup in pickups:
            if pickup['cancelled']:
                continue
            
            # Verify they actually walked away from the slot during their track
            if pickup['max_dist'] <= 160.0:
                pickup['cancelled'] = True
                continue
            
            slot_idx = pickup['slot_idx']
            # Check if the slot was ever occupied again from the pickup frame until the very end of the video
            was_occupied_ever_again = False
            for f in range(pickup['frame_pickup'], len(frames_detections)):
                if slot_occupancy[f][slot_idx]:
                    was_occupied_ever_again = True
                    break
            
            if was_occupied_ever_again:
                pickup['cancelled'] = True
                continue
            
            # Verify that the slot WAS actually occupied at some point before the pickup
            was_occupied_before = False
            for f in range(0, pickup['frame_pickup']):
                if slot_occupancy[f][slot_idx]:
                    was_occupied_before = True
                    break
            
            if not was_occupied_before:
                pickup['cancelled'] = True
                continue
            
            pickup['confirmed'] = True
            reason = "Item concealed in bag" if pickup['has_bag'] else "Item concealed in pocket"
            
            start_time = max(0, (pickup['frame_pickup'] / fps) - 2.0)
            end_time = min(duration, (last_seen / fps) + 2.0)
            
            suspicious_events.append({
                'track_id': track_id,
                'start_time': start_time,
                'end_time': end_time,
                'reason': reason
            })
            
    return suspicious_events

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    correct = 0
    total = len(cache)
    
    results = []
    for i, (key, video_data) in enumerate(sorted(cache.items()), 1):
        label = "shoplifting" if key.startswith("shoplifting/") else "normal"
        events = process_video(video_data)
        
        expected_detect = label == "shoplifting"
        detected = len(events) > 0
        ok = detected == expected_detect
        if ok:
            correct += 1
        results.append((key, label, len(events), ok))
        
        status = "OK" if ok else "FAIL"
        print(f"[{i}/{total}] {status} {key}: {len(events)} events (expected {'detect' if expected_detect else 'none'})")
        
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Total Correct: {correct}/{total} ({100 * correct / total:.2f}%)")
    
    normals = [r for r in results if r[1] == "normal"]
    shops = [r for r in results if r[1] == "shoplifting"]
    
    print(f"Normal Correct: {sum(1 for r in normals if r[3])}/{len(normals)}")
    print(f"Shoplifting Correct: {sum(1 for r in shops if r[3])}/{len(shops)}")

if __name__ == "__main__":
    main()
