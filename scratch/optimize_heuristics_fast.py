#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def precompute_all(cache):
    print("Precomputing static slots and raw occupancy counts for all videos...")
    precomputed = {}
    for key, video_data in cache.items():
        frames_detections = video_data['detections']
        fps = video_data['metadata']['fps']
        duration = video_data['metadata']['duration']
        
        # Cluster static slots
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
        
        # Precompute raw item counts
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
            
        precomputed[key] = {
            'shelf_slots': shelf_slots,
            'slot_occupancy_counts': slot_occupancy_counts,
            'fps': fps,
            'duration': duration,
            'frames_detections': frames_detections
        }
    print("Precomputation finished!")
    return precomputed

def evaluate_params(cache, precomputed, filter_size, interaction_radius, walk_away_dist, drop_window, return_margin, after_margin, end_buffer):
    correct = 0
    
    # Pre-smooth counts for all videos to avoid smoothing inside the tracking loop
    smoothed_occupancy_all = {}
    for key, data in precomputed.items():
        slot_occupancy_counts = data['slot_occupancy_counts']
        shelf_slots = data['shelf_slots']
        
        smoothed_occupancy_counts = []
        if filter_size > 1:
            window_half = filter_size // 2
            for frame_idx in range(len(slot_occupancy_counts)):
                frame_smoothed = []
                for slot_idx in range(len(shelf_slots)):
                    window = []
                    for f in range(max(0, frame_idx - window_half), min(len(slot_occupancy_counts), frame_idx + window_half + 1)):
                        window.append(slot_occupancy_counts[f][slot_idx])
                    frame_smoothed.append(sorted(window)[len(window)//2])
                smoothed_occupancy_counts.append(frame_smoothed)
        else:
            smoothed_occupancy_counts = slot_occupancy_counts
        smoothed_occupancy_all[key] = smoothed_occupancy_counts

    for key, data in precomputed.items():
        label = "shoplifting" if key.startswith("shoplifting/") else "normal"
        frames_detections = data['frames_detections']
        fps = data['fps']
        duration = data['duration']
        shelf_slots = data['shelf_slots']
        smoothed_occupancy_counts = smoothed_occupancy_all[key]
        
        tracker = ByteTracker()
        potential_pickups = defaultdict(list)
        person_last_seen_frame = {}
        person_first_seen_frame = {}
        
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
                
                for slot_idx, slot in enumerate(shelf_slots):
                    scx, scy = slot['center']
                    is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                    dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                    
                    if is_inside or dist < interaction_radius:
                        existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                        
                        # Reset pickup if count returns to baseline capacity (item was put back)
                        if existing and current_counts[slot_idx] >= existing['baseline_capacity'] - return_margin:
                            potential_pickups[track_id] = [p for p in potential_pickups[track_id] if p['slot_idx'] != slot_idx]
                            existing = None
                            
                        # Drop check: did the count drop recently?
                        past_window = range(max(0, frame_num - drop_window), frame_num)
                        past_max = max([smoothed_occupancy_counts[f][slot_idx] for f in past_window]) if past_window else current_counts[slot_idx]
                        
                        if current_counts[slot_idx] < past_max and past_max > 0:
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
                                    'baseline_capacity': past_max,
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
                    max_dist_after > walk_away_dist or 
                    last_seen < len(frames_detections) - 5 or 
                    last_dist > 80.0
                )
                
                # Return check
                was_returned = False
                for f in range(pickup['frame_pickup'] + 2, len(frames_detections)):
                    if smoothed_occupancy_counts[f][slot_idx] >= baseline_capacity - return_margin:
                        was_returned = True
                        break
                
                # Calculate avg count after
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
                if pickup['frame_pickup'] >= len(frames_detections) - end_buffer:
                    reasons.append("pickup too close to end of video")
                if not walked_away:
                    reasons.append("not walked away")
                if was_returned:
                    reasons.append("returned")
                if avg_count_after > baseline_capacity - after_margin:
                    reasons.append("avg_count_after > baseline")
                    
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
                    
        expected_detect = label == "shoplifting"
        detected = len(suspicious_events) > 0
        if detected == expected_detect:
            correct += 1
            
    return correct

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    precomputed = precompute_all(cache)
    print("Starting fast grid search...")
    
    best_correct = 0
    best_params = None
    
    # Highly targeted search parameters for rapid exploration
    for filter_size in [5, 11, 25]:
        for interaction_radius in [100.0, 130.0, 150.0]:
            for walk_away_dist in [120.0, 150.0, 180.0]:
                for drop_window in [15, 25]:
                    for return_margin in [0.0, 0.2]:
                        for after_margin in [0.3, 0.4, 0.5]:
                            for end_buffer in [15, 30]:
                                correct = evaluate_params(
                                    cache, precomputed, filter_size, interaction_radius, walk_away_dist,
                                    drop_window, return_margin, after_margin, end_buffer
                                )
                                pct = 100 * correct / len(cache)
                                if correct > best_correct:
                                    best_correct = correct
                                    best_params = (filter_size, interaction_radius, walk_away_dist, drop_window, return_margin, after_margin, end_buffer)
                                    print(f"*** NEW BEST: {correct}/{len(cache)} ({pct:.2f}%) -> filter={filter_size}, radius={interaction_radius}, walk={walk_away_dist}, drop={drop_window}, ret={return_margin}, after={after_margin}, end={end_buffer}")
                                    
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print("="*50)
    print(f"Accuracy: {best_correct}/{len(cache)} ({100*best_correct/len(cache):.2f}%)")
    print(f"filter_size: {best_params[0]}")
    print(f"interaction_radius: {best_params[1]}")
    print(f"walk_away_dist: {best_params[2]}")
    print(f"drop_window: {best_params[3]}")
    print(f"return_margin: {best_params[4]}")
    print(f"after_margin: {best_params[5]}")
    print(f"end_buffer: {best_params[6]}")

if __name__ == "__main__":
    main()
