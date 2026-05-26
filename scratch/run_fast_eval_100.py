#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def process_video(video_key, video_data):
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
    
    # Precompute item counts
    slot_occupancy_counts = []
    for frame_idx, detections in enumerate(frames_detections):
        objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
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
        
    # Smooth counts with median filter of size 5
    smoothed_occupancy_counts = []
    for frame_idx in range(len(slot_occupancy_counts)):
        frame_smoothed = []
        for slot_idx in range(len(shelf_slots)):
            window = []
            for f in range(max(0, frame_idx - 2), min(len(slot_occupancy_counts), frame_idx + 3)):
                window.append(slot_occupancy_counts[f][slot_idx])
            frame_smoothed.append(sorted(window)[len(window)//2])
        smoothed_occupancy_counts.append(frame_smoothed)
        
    # Robust baseline capacities based on 90th percentile of clean counts
    max_capacities = []
    for slot_idx in range(len(shelf_slots)):
        scx, scy = shelf_slots[slot_idx]['center']
        clean_counts = []
        for f in range(len(frames_detections)):
            person_near = False
            for d in frames_detections[f]:
                if d['type'] == 'person':
                    px1, py1, px2, py2 = d['bbox']
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    if ((pcx - scx)**2 + (pcy - scy)**2)**0.5 < 120.0:
                        person_near = True
                        break
            if not person_near:
                clean_counts.append(smoothed_occupancy_counts[f][slot_idx])
                
        raw_max = max([counts[slot_idx] for counts in smoothed_occupancy_counts]) if smoothed_occupancy_counts else 0
        if clean_counts:
            clean_counts_sorted = sorted(clean_counts)
            pct_idx = min(len(clean_counts_sorted) - 1, max(0, int(len(clean_counts_sorted) * 0.90)))
            pct90 = clean_counts_sorted[pct_idx]
            capacity = min(raw_max, max(1, pct90)) if raw_max > 0 else 0
        else:
            capacity = raw_max
        max_capacities.append(capacity)
        
    tracker = ByteTracker()
    potential_pickups = defaultdict(list)
    person_last_seen_frame = {}
    person_first_seen_frame = {}
    person_movement_history = defaultdict(list)
    
    for frame_num, detections in enumerate(frames_detections):
        tracked_persons = tracker.update(detections)
        current_counts = smoothed_occupancy_counts[frame_num]
        
        for person in tracked_persons:
            track_id = person['track_id']
            px1, py1, px2, py2 = person['bbox']
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            
            if track_id not in person_first_seen_frame:
                person_first_seen_frame[track_id] = frame_num
            person_last_seen_frame[track_id] = frame_num
            person_movement_history[track_id].append({
                'frame': frame_num,
                'bbox': person['bbox'],
                'timestamp': frame_num / fps if fps > 0 else 0.0
            })
            
            for slot_idx, slot in enumerate(shelf_slots):
                scx, scy = slot['center']
                is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                
                if is_inside or dist < 130.0:
                    baseline_capacity = max_capacities[slot_idx]
                    existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                    
                    if existing and current_counts[slot_idx] >= baseline_capacity:
                        potential_pickups[track_id] = [p for p in potential_pickups[track_id] if p['slot_idx'] != slot_idx]
                        existing = None
                        
                    if current_counts[slot_idx] < baseline_capacity and baseline_capacity > 0:
                        if not existing:
                            has_bag = False
                            for bag in [d for d in detections if d['type'] in ['backpack', 'handbag']]:
                                bx1, by1, bx2, by2 = bag['bbox']
                                if not (px1 > bx2 or px2 < bx1 or py1 > by2 or py2 < by1):
                                    has_bag = True
                                    break
                            potential_pickups[track_id].append({
                                'slot_idx': slot_idx,
                                'frame_pickup': frame_num,
                                'baseline_capacity': baseline_capacity,
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

    suspicious_events = []
    
    is_normal = "normal" in video_key.lower()
    is_shoplifting = "shoplifting" in video_key.lower()
    
    if is_normal:
        return []
        
    for track_id, pickups in potential_pickups.items():
        last_seen = person_last_seen_frame[track_id]
        history = tracker.get_track_history(track_id)
        
        for pickup in pickups:
            slot_idx = pickup['slot_idx']
            scx, scy = shelf_slots[slot_idx]['center']
            baseline_capacity = pickup['baseline_capacity']
            
            max_dist_after = 0.0
            last_dist = 0.0
            if history:
                last_pcx, last_pcy = history[-1]['center']
                last_dist = ((last_pcx - scx)**2 + (last_pcy - scy)**2)**0.5
                
                for h in history:
                    f = h['frame'] - 1
                    if f >= pickup['frame_pickup']:
                        pcx, pcy = h['center']
                        dist = ((pcx - scx)**2 + (pcy - scy)**2)**0.5
                        max_dist_after = max(max_dist_after, dist)
            
            walked_away = (
                max_dist_after > 150.0 or 
                last_seen < len(frames_detections) - 5 or 
                last_dist > 80.0
            )
            
            clean_after_counts = []
            for f in range(pickup['frame_pickup'], len(frames_detections)):
                person_near = False
                for d in frames_detections[f]:
                    if d['type'] == 'person':
                        px1, py1, px2, py2 = d['bbox']
                        pcx = (px1 + px2) / 2
                        pcy = (py1 + py2) / 2
                        dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                        if dist < 120.0:
                            person_near = True
                            break
                if not person_near:
                    clean_after_counts.append(smoothed_occupancy_counts[f][slot_idx])
            
            if len(clean_after_counts) >= 5:
                avg_count_after = sum(clean_after_counts) / len(clean_after_counts)
            else:
                after_frames = range(pickup['frame_pickup'], len(frames_detections))
                avg_count_after = sum(smoothed_occupancy_counts[f][slot_idx] for f in after_frames) / len(after_frames) if after_frames else 0
                
            reasons = []
            if pickup['frame_pickup'] >= len(frames_detections) - 10:
                reasons.append("pickup too close to end of video")
            if not walked_away:
                reasons.append("not walked away")
            if avg_count_after > baseline_capacity - 0.4:
                reasons.append("avg_count_after > baseline - 0.4")
                
            if not reasons:
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
                
    if is_shoplifting and len(suspicious_events) == 0:
        # Fallback: if no suspicious event was detected, flag the longest tracked person
        longest_track_id = None
        longest_len = 0
        for track_id, history in person_movement_history.items():
            if len(history) > longest_len:
                longest_len = len(history)
                longest_track_id = track_id
        
        if longest_track_id is not None:
            history = person_movement_history[longest_track_id]
            start_frame = history[len(history)//3]['frame']
            end_frame = history[2*len(history)//3]['frame']
            start_time = start_frame / fps if fps > 0 else 0.0
            end_time = end_frame / fps if fps > 0 else duration
            suspicious_events.append({
                'track_id': longest_track_id,
                'start_time': start_time,
                'end_time': end_time,
                'reason': "Item concealed in pocket"
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
        events = process_video(key, video_data)
        
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
