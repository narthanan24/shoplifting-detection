#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tracker import ByteTracker

def check_overlap(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)

def analyze_item_presence_after_pickup():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    print("Analyzing item presence in person bboxes after pickup...")
    
    results = []
    
    for key, video_data in cache.items():
        is_shoplifting = key.startswith("shoplifting/")
        frames_detections = video_data['detections']
        
        # Identify shelf slots
        raw_shelf_detections = []
        for frame_num, detections in enumerate(frames_detections):
            persons = [d for d in detections if d['type'] == 'person']
            objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
            
            for obj in objects:
                overlaps = False
                for person in persons:
                    if check_overlap(person['bbox'], obj['bbox']):
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
        if not shelf_slots:
            continue
            
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
            
        max_capacities = [max([counts[slot_idx] for counts in slot_occupancy_counts]) for slot_idx in range(len(shelf_slots))]
        
        tracker = ByteTracker()
        potential_pickups = defaultdict(list)
        
        for frame_num, detections in enumerate(frames_detections):
            tracked_persons = tracker.update(detections)
            current_counts = slot_occupancy_counts[frame_num]
            
            for person in tracked_persons:
                track_id = person['track_id']
                px1, py1, px2, py2 = person['bbox']
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                
                for slot_idx, slot in enumerate(shelf_slots):
                    scx, scy = slot['center']
                    is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                    dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                    
                    if is_inside or dist < 130.0:
                        if current_counts[slot_idx] < max_capacities[slot_idx]:
                            existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                            if not existing:
                                potential_pickups[track_id].append({
                                    'slot_idx': slot_idx,
                                    'frame_pickup': frame_num,
                                    'max_dist': dist
                                })
                                
        # For each registered pickup, let's analyze the frames after pickup
        for track_id, pickups in potential_pickups.items():
            # Find the track's full history of frames
            history = tracker.get_track_history(track_id)
            if not history:
                continue
                
            track_frames = {h['frame']: h for h in history}
            
            for pickup in pickups:
                f_pickup = pickup['frame_pickup']
                slot_idx = pickup['slot_idx']
                
                # Frames where this person is tracked *after* the pickup
                after_frames = [f for f in track_frames.keys() if f > f_pickup]
                if not after_frames:
                    continue
                    
                overlap_count = 0
                total_checked = 0
                
                for f in after_frames:
                    person_box = track_frames[f]['bbox']
                    # Check if there is any item/bottle detection in this frame overlapping the person
                    frame_items = [d for d in frames_detections[f - 1] if d['type'] in ['bottle', 'item']]
                    
                    has_item_overlap = False
                    for item in frame_items:
                        # Ensure the item is not close to any shelf slot (i.e. it is being carried, not on shelf)
                        on_shelf = False
                        icx = (item['bbox'][0] + item['bbox'][2]) / 2
                        icy = (item['bbox'][1] + item['bbox'][3]) / 2
                        for s in shelf_slots:
                            scx, scy = s['center']
                            if ((icx - scx)**2 + (icy - scy)**2)**0.5 < 45.0:
                                on_shelf = True
                                break
                                
                        if not on_shelf and check_overlap(person_box, item['bbox']):
                            has_item_overlap = True
                            break
                            
                    if has_item_overlap:
                        overlap_count += 1
                    total_checked += 1
                    
                overlap_ratio = overlap_count / total_checked if total_checked > 0 else 0
                results.append({
                    'key': key,
                    'is_shoplifting': is_shoplifting,
                    'track_id': track_id,
                    'slot_idx': slot_idx,
                    'overlap_ratio': overlap_ratio,
                    'total_frames_after': total_checked
                })
                
    # Summary of results
    shop_ratios = [r['overlap_ratio'] for r in results if r['is_shoplifting']]
    norm_ratios = [r['overlap_ratio'] for r in results if not r['is_shoplifting']]
    
    print("\nOverlap ratio (fraction of frames where person carries a visible item after pickup):")
    print(f"Normal videos: {len(norm_ratios)} events")
    if norm_ratios:
        print(f"  Min: {min(norm_ratios):.2f}, Max: {max(norm_ratios):.2f}, Avg: {sum(norm_ratios)/len(norm_ratios):.2f}")
    print(f"Shoplifting videos: {len(shop_ratios)} events")
    if shop_ratios:
        print(f"  Min: {min(shop_ratios):.2f}, Max: {max(shop_ratios):.2f}, Avg: {sum(shop_ratios)/len(shop_ratios):.2f}")

if __name__ == "__main__":
    analyze_item_presence_after_pickup()
