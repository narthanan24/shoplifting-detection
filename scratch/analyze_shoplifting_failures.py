#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def process_video_debug(video_data):
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
        
    # Smooth counts with median filter of size 25
    smoothed_occupancy_counts = []
    for frame_idx in range(len(slot_occupancy_counts)):
        frame_smoothed = []
        for slot_idx in range(len(shelf_slots)):
            window = []
            for f in range(max(0, frame_idx - 12), min(len(slot_occupancy_counts), frame_idx + 13)):
                window.append(slot_occupancy_counts[f][slot_idx])
            frame_smoothed.append(sorted(window)[len(window)//2])
        smoothed_occupancy_counts.append(frame_smoothed)
        
    tracker = ByteTracker()
    potential_pickups = defaultdict(list)
    person_last_seen_frame = {}
    person_first_seen_frame = {}
    
    person_slot_approach = defaultdict(dict)
    
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
                
                if is_inside or dist < 130.0:
                    if slot_idx not in person_slot_approach[track_id]:
                        past_window = range(max(0, frame_num - 25), frame_num + 1)
                        approach_baseline = max([smoothed_occupancy_counts[f][slot_idx] for f in past_window])
                        person_slot_approach[track_id][slot_idx] = approach_baseline
                        
                    baseline_capacity = person_slot_approach[track_id][slot_idx]
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
                else:
                    if dist >= 160.0:
                        person_slot_approach[track_id].pop(slot_idx, None)
                            
            for pickup in potential_pickups[track_id]:
                if pickup['cancelled'] or pickup['confirmed']:
                    continue
                slot_idx = pickup['slot_idx']
                scx, scy = shelf_slots[slot_idx]['center']
                dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                pickup['max_dist'] = max(pickup['max_dist'], dist)

    results = []
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
            
            was_returned = False
            for f in range(pickup['frame_pickup'] + 2, len(frames_detections)):
                if smoothed_occupancy_counts[f][slot_idx] >= baseline_capacity:
                    was_returned = True
                    break
            
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
            if pickup['frame_pickup'] >= len(frames_detections) - 30:
                reasons.append("pickup close to end of video")
            if not walked_away:
                reasons.append(f"not walked away (max_dist_after={max_dist_after:.1f}, last_seen={last_seen}/{len(frames_detections)}, last_dist={last_dist:.1f})")
            if was_returned:
                reasons.append("returned")
            if avg_count_after > baseline_capacity - 0.4:
                reasons.append(f"avg_count_after ({avg_count_after:.2f}) > baseline ({baseline_capacity}) - 0.4")
                
            results.append({
                'track_id': track_id,
                'slot_idx': slot_idx,
                'reasons': reasons,
                'pickup_frame': pickup['frame_pickup'],
                'total_frames': len(frames_detections)
            })
            
    return results

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    print("Analyzing missed shoplifting videos...")
    no_pickup = 0
    total_missed = 0
    cancellation_counts = defaultdict(int)
    
    for key, video_data in sorted(cache.items()):
        if not key.startswith("shoplifting/"):
            continue
            
        pickups = process_video_debug(video_data)
        confirmed = [p for p in pickups if not p['reasons']]
        
        if not confirmed:
            total_missed += 1
            if not pickups:
                no_pickup += 1
                print(f"FAIL {key}: NO pickups registered at all!")
            else:
                print(f"FAIL {key}: {len(pickups)} pickups registered but all cancelled:")
                for p in pickups:
                    print(f"  Person {p['track_id']}, Slot {p['slot_idx']}, frame {p['pickup_frame']}/{p['total_frames']}, cancelled due to: {', '.join(p['reasons'])}")
                    for r in p['reasons']:
                        cancellation_counts[r] += 1
                        
    print(f"\nSummary:")
    print(f"Total Missed: {total_missed}")
    print(f"  No Pickups Registered: {no_pickup}")
    print("Cancellation reasons distribution:")
    for r, count in sorted(cancellation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r}: {count}")

if __name__ == "__main__":
    main()
