#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def diagnose_video(video_key, cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    if video_key not in cache:
        print(f"Video {video_key} not in cache.")
        return
        
    video_data = cache[video_key]
    metadata = video_data['metadata']
    frames_detections = video_data['detections']
    fps = metadata['fps']
    duration = metadata['duration']
    
    print(f"--- Diagnosing {video_key} ---")
    print(f"FPS: {fps}, Duration: {duration:.2f}s, Total Frames: {len(frames_detections)}")
    
    # 1. Identify static shelf slots
    raw_shelf_detections = []
    for frame_num, detections in enumerate(frames_detections):
        persons = [d for d in detections if d['type'] == 'person']
        objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
        
        for obj in objects:
            overlaps_person = False
            for person in persons:
                # check overlap
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
    print(f"Detected {len(shelf_slots)} shelf slots (out of {len(slots)} total candidate clusters):")
    for idx, slot in enumerate(shelf_slots):
        print(f"  Slot {idx}: center={slot['center']}, occurrences={slot['count']}")
        
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
        
    max_capacities = []
    for slot_idx in range(len(shelf_slots)):
        max_c = max([counts[slot_idx] for counts in slot_occupancy_counts]) if slot_occupancy_counts else 0
        max_capacities.append(max_c)
        print(f"  Slot {slot_idx} Max Capacity: {max_c}")
        
    # Tracker
    tracker = ByteTracker()
    potential_pickups = defaultdict(list)
    person_last_seen_frame = {}
    person_first_seen_frame = {}
    
    for frame_num, detections in enumerate(frames_detections):
        persons = [d for d in detections if d['type'] == 'person']
        bags = [d for d in detections if d['type'] in ['backpack', 'handbag']]
        tracked_persons = tracker.update(detections)
        current_counts = slot_occupancy_counts[frame_num]
        
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
                    if current_counts[slot_idx] < max_capacities[slot_idx]:
                        existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                        if not existing:
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
                                'confirmed': False,
                                'counts_history_around_pickup': []
                            })
                            print(f"Frame {frame_num} ({frame_num/fps:.1f}s): Person {track_id} registered pickup from Slot {slot_idx}. current_count={current_counts[slot_idx]} (max={max_capacities[slot_idx]}), dist={dist:.1f}, has_bag={has_bag}")
                            
            for pickup in potential_pickups[track_id]:
                if pickup['cancelled'] or pickup['confirmed']:
                    continue
                slot_idx = pickup['slot_idx']
                scx, scy = shelf_slots[slot_idx]['center']
                dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                pickup['max_dist'] = max(pickup['max_dist'], dist)
                # record recent count history around pickup
                pickup['counts_history_around_pickup'].append((frame_num, current_counts[slot_idx], dist))

    # Evaluate post-video
    print("\n--- Post-Video Analysis ---")
    for track_id, pickups in potential_pickups.items():
        last_seen = person_last_seen_frame[track_id]
        for pickup in pickups:
            slot_idx = pickup['slot_idx']
            baseline_count = max_capacities[slot_idx]
            
            # Check clean frames after pickup
            clean_after_counts = []
            clean_frame_details = []
            for f in range(pickup['frame_pickup'], len(frames_detections)):
                person_near = False
                for d in frames_detections[f]:
                    if d['type'] == 'person':
                        px1, py1, px2, py2 = d['bbox']
                        pcx = (px1 + px2) / 2
                        pcy = (py1 + py2) / 2
                        scx, scy = shelf_slots[slot_idx]['center']
                        dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                        if dist < 120.0:
                            person_near = True
                            break
                if not person_near:
                    clean_after_counts.append(slot_occupancy_counts[f][slot_idx])
                    clean_frame_details.append((f, slot_occupancy_counts[f][slot_idx]))
            
            if len(clean_after_counts) >= 5:
                avg_count_after = sum(clean_after_counts) / len(clean_after_counts)
                src = "clean frames"
            else:
                after_frames = range(pickup['frame_pickup'], len(frames_detections))
                avg_count_after = sum(slot_occupancy_counts[f][slot_idx] for f in after_frames) / len(after_frames) if after_frames else 0
                src = "all frames fallback"
                
            # Was full before?
            was_full_before = False
            for f in range(0, pickup['frame_pickup']):
                if slot_occupancy_counts[f][slot_idx] >= max_capacities[slot_idx]:
                    was_full_before = True
                    break
                    
            print(f"Person {track_id}, Slot {slot_idx}:")
            print(f"  Frame Pickup: {pickup['frame_pickup']} ({pickup['frame_pickup']/fps:.1f}s)")
            print(f"  Max Distance: {pickup['max_dist']:.1f} (needs > 160.0)")
            print(f"  Was Full Before: {was_full_before}")
            print(f"  Clean after frames count: {len(clean_after_counts)} (details of first 5: {clean_frame_details[:5]})")
            print(f"  Avg count after: {avg_count_after:.2f} (baseline: {baseline_count}, threshold limit: {baseline_count - 0.4:.2f})")
            
            # Reasons for cancellation
            reasons = []
            if pickup['max_dist'] <= 160.0:
                reasons.append("max_dist <= 160")
            if avg_count_after > baseline_count - 0.4:
                reasons.append(f"avg_count_after ({avg_count_after:.2f}) > baseline_count - 0.4 ({baseline_count - 0.4:.2f})")
            if not was_full_before:
                reasons.append("not full before")
                
            if reasons:
                print(f"  STATUS: CANCELLED due to {', '.join(reasons)}")
            else:
                print(f"  STATUS: CONFIRMED THEFT")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: diagnose_video.py <video_key>")
        sys.exit(1)
    diagnose_video(sys.argv[1], "evaluation_results/detections_cache.pkl")
