#!/usr/bin/env python3
"""Fast evaluation using cached YOLO detections."""
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent))
from tracker import ByteTracker
from utils import merge_overlapping_timestamps


class SimulatedDetector:
    def __init__(self, 
                 shelf_region=None,
                 time_near_shelf_threshold=15.0,
                 item_disappear_buffer=3.0,
                 exit_buffer=5.0):
        self.shelf_region = shelf_region
        self.time_near_shelf_threshold = time_near_shelf_threshold
        self.item_disappear_buffer = item_disappear_buffer
        self.exit_buffer = exit_buffer
        
        # State tracking per person
        self.person_states = defaultdict(dict)
        self.suspicious_events = []
        self.person_movement_history = defaultdict(list)
        self.person_item_interactions = defaultdict(list)

    def _is_near_shelf(self, bbox) -> bool:
        if self.shelf_region is None:
            return True
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = self.shelf_region
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        margin = 50
        return (sx1 - margin <= center_x <= sx2 + margin and 
                sy1 - margin <= center_y <= sy2 + margin)

    def _calculate_distance(self, bbox1, bbox2) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    def _find_nearby_items(self, person_bbox, items, threshold=200.0):
        nearby = []
        for item in items:
            distance = self._calculate_distance(person_bbox, item['bbox'])
            if distance < threshold:
                nearby.append(item)
        return nearby

    def _check_overlap(self, bbox1, bbox2) -> bool:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)

    def process_cached_video(self, video_data) -> list:
        metadata = video_data['metadata']
        frames_detections = video_data['detections']
        fps = metadata['fps']
        duration = metadata['duration']
        
        # Step 1: Identify static shelf slots across all frames
        # We only consider item objects that are NOT overlapping with any person's bounding box
        raw_shelf_detections = []
        for frame_num, detections in enumerate(frames_detections):
            persons = [d for d in detections if d['type'] == 'person']
            objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
            
            for obj in objects:
                overlaps_person = False
                for person in persons:
                    if self._check_overlap(person['bbox'], obj['bbox']):
                        overlaps_person = True
                        break
                if not overlaps_person:
                    ox1, oy1, ox2, oy2 = obj['bbox']
                    cx = (ox1 + ox2) / 2
                    cy = (oy1 + oy2) / 2
                    raw_shelf_detections.append((cx, cy))
        
        # Cluster the raw shelf detections to find static slots
        slots = []
        for cx, cy in raw_shelf_detections:
            found = False
            for slot in slots:
                scx, scy = slot['center']
                dist = ((cx - scx) ** 2 + (cy - scy) ** 2) ** 0.5
                if dist < 30.0:  # 30 pixels matching radius
                    n = slot['count']
                    slot['center'] = (
                        (scx * n + cx) / (n + 1),
                        (scy * n + cy) / (n + 1)
                    )
                    slot['count'] += 1
                    found = True
                    break
            if not found:
                slots.append({
                    'center': (cx, cy),
                    'count': 1
                })
        
        # Keep slots with high confidence (detected at least 15 times throughout the video)
        shelf_slots = [s for s in slots if s['count'] >= 15]
        
        tracker = ByteTracker()
        
        # Precompute item counts near each slot center for all frames
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
            
        # Robust baseline capacities based on 90th percentile of counts when no person is near this slot
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
                    clean_counts.append(slot_occupancy_counts[f][slot_idx])
            
            raw_max = max([counts[slot_idx] for counts in slot_occupancy_counts]) if slot_occupancy_counts else 0
            if clean_counts:
                # 90th percentile
                clean_counts_sorted = sorted(clean_counts)
                pct_idx = min(len(clean_counts_sorted) - 1, max(0, int(len(clean_counts_sorted) * 0.90)))
                pct90 = clean_counts_sorted[pct_idx]
                capacity = min(raw_max, max(1, pct90)) if raw_max > 0 else 0
            else:
                capacity = raw_max
            max_capacities.append(capacity)
        
        # We track state of potential pickups: person_id -> list of dicts:
        # { 'slot_idx': int, 'frame_pickup': int, 'max_dist': float, 'cancelled': bool, 'has_bag': bool, 'confirmed': bool }
        potential_pickups = defaultdict(list)
        
        person_last_seen_frame = {}
        person_first_seen_frame = {}
        
        for frame_num, detections in enumerate(frames_detections):
            timestamp = frame_num / fps if fps > 0 else 0.0
            
            # Split detections
            persons = [d for d in detections if d['type'] == 'person']
            objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
            bags = [d for d in detections if d['type'] in ['backpack', 'handbag']]
            
            # Track persons
            tracked_persons = tracker.update(detections)
            current_counts = slot_occupancy_counts[frame_num]
            
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
                    
                    # Interacting if slot center is inside person's bounding box (with 20px margin) OR center distance is small
                    is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                    dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                    
                    if is_inside or dist < 130.0:
                        # Pickup if the current count is less than the max capacity for this slot
                        if current_counts[slot_idx] < max_capacities[slot_idx]:
                            existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                            if not existing:
                                # Check if they carry a bag
                                has_bag = False
                                for bag in bags:
                                    if self._check_overlap(person['bbox'], bag['bbox']):
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
                
                # For each potential pickup of this person, track maximum distance reached from the slot
                for pickup in potential_pickups[track_id]:
                    if pickup['cancelled'] or pickup['confirmed']:
                        continue
                        
                    slot_idx = pickup['slot_idx']
                    scx, scy = shelf_slots[slot_idx]['center']
                    dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                    pickup['max_dist'] = max(pickup['max_dist'], dist)
        
        # Post-video analysis: verify and flag events
        for track_id, pickups in potential_pickups.items():
            last_seen = person_last_seen_frame[track_id]
            
            # Get last seen position
            history = tracker.get_track_history(track_id)
            if history:
                last_pcx, last_pcy = history[-1]['center']
            else:
                last_pcx, last_pcy = 0.0, 0.0
            
            for pickup in pickups:
                if pickup['cancelled']:
                    continue
                
                slot_idx = pickup['slot_idx']
                scx, scy = shelf_slots[slot_idx]['center']
                baseline_count = max_capacities[slot_idx]
                last_dist = ((last_pcx - scx)**2 + (last_pcy - scy)**2)**0.5
                
                # 1. Walk-away / exit check:
                # The person must walk away from the slot, OR exit the frame early, OR be moving away at the end
                walked_away = (
                    pickup['max_dist'] > 150.0 or 
                    last_seen < len(frames_detections) - 5 or 
                    last_dist > 80.0
                )
                
                if not walked_away:
                    pickup['cancelled'] = True
                    continue
                
                # 2. Return Check:
                # If the count in the slot ever returns to >= baseline_count after pickup, they put it back or it was occlusion.
                was_returned = False
                for f in range(pickup['frame_pickup'] + 2, len(frames_detections)):
                    if slot_occupancy_counts[f][slot_idx] >= baseline_count:
                        was_returned = True
                        break
                
                if was_returned:
                    pickup['cancelled'] = True
                    continue
                
                # 3. Calculate average count in "clean frames" (where no person is near this slot) after the pickup
                clean_after_counts = []
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
                
                if len(clean_after_counts) >= 5:
                    avg_count_after = sum(clean_after_counts) / len(clean_after_counts)
                else:
                    # Fallback to all frames after pickup if there are too few clean frames
                    after_frames = range(pickup['frame_pickup'], len(frames_detections))
                    avg_count_after = sum(slot_occupancy_counts[f][slot_idx] for f in after_frames) / len(after_frames) if after_frames else 0
                
                # If the average count has not dropped significantly compared to baseline, cancel
                if avg_count_after > baseline_count - 0.4:
                    pickup['cancelled'] = True
                    continue
                
                # 4. Verify that the slot WAS actually at baseline capacity at some point before the pickup
                was_full_before = False
                for f in range(0, pickup['frame_pickup']):
                    if slot_occupancy_counts[f][slot_idx] >= baseline_count:
                        was_full_before = True
                        break
                
                if not was_full_before:
                    pickup['cancelled'] = True
                    continue
                
                # If all checks pass, it's a confirmed theft
                pickup['confirmed'] = True
                
                # Determine reason
                reason = "Item concealed in bag" if pickup['has_bag'] else "Item concealed in pocket"
                
                start_time = max(0, (pickup['frame_pickup'] / fps) - 2.0)
                end_time = min(duration, (last_seen / fps) + 2.0)
                
                self.suspicious_events.append({
                    'track_id': track_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': reason
                })
        
        # Merge overlapping events
        if self.suspicious_events:
            intervals = [(e['start_time'], e['end_time']) for e in self.suspicious_events]
            merged_intervals = merge_overlapping_timestamps(intervals, merge_threshold=1.0)
            
            merged_events = []
            for start, end in merged_intervals:
                overlapping = [e for e in self.suspicious_events 
                             if not (e['end_time'] < start or e['start_time'] > end)]
                if overlapping:
                    track_id = overlapping[0]['track_id']
                    reasons = ', '.join(set(e['reason'] for e in overlapping))
                    merged_events.append({
                        'track_id': track_id,
                        'start_time': start,
                        'end_time': end,
                        'reason': reasons
                    })
            self.suspicious_events = merged_events
            
        # Select at most one event (the one with the longest duration)
        if len(self.suspicious_events) > 1:
            self.suspicious_events = [max(self.suspicious_events, key=lambda e: e['end_time'] - e['start_time'])]
            
        return self.suspicious_events


def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    if not cache_path.exists():
        print(f"Error: Cache file {cache_path} does not exist. Please run cache_detections.py first.")
        return

    print("Loading detections cache...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded detections for {len(cache)} videos.")

    # We can pass limit argument to evaluate only subset (e.g. 10 or 20)
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    results = []
    
    # Sort keys for consistent ordering
    keys = sorted(cache.keys())
    
    # If a limit is specified, take first N normal and first N shoplifting videos
    if limit is not None:
        normal_keys = [k for k in keys if k.startswith("normal/")]
        shop_keys = [k for k in keys if k.startswith("shoplifting/")]
        keys = normal_keys[:limit] + shop_keys[:limit]

    total = len(keys)
    correct = 0

    for i, key in enumerate(keys, 1):
        label = "shoplifting" if key.startswith("shoplifting/") else "normal"
        video_data = cache[key]
        
        detector = SimulatedDetector(time_near_shelf_threshold=15.0)
        events = detector.process_cached_video(video_data)
        
        expected_detect = label == "shoplifting"
        detected = len(events) > 0
        ok = detected == expected_detect
        if ok:
            correct += 1
            
        results.append((key, label, len(events), ok, events))
        
        status = "OK" if ok else "FAIL"
        reasons = [e['reason'] for e in events]
        reasons_str = f" ({', '.join(reasons)})" if reasons else ""
        print(f"[{i}/{total}] {status} {key}: {len(events)} event(s) expected {'detect' if expected_detect else 'none'}{reasons_str}")

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total: {correct}/{total} ({100 * correct / total:.1f}%)")
    
    normal_results = [r for r in results if r[1] == "normal"]
    shop_results = [r for r in results if r[1] == "shoplifting"]
    
    normal_correct = sum(1 for r in normal_results if r[3])
    shop_correct = sum(1 for r in shop_results if r[3])
    
    print(f"Normal: {normal_correct}/{len(normal_results)} correct")
    print(f"Shoplifting: {shop_correct}/{len(shop_results)} correct")
    
    failures = [r for r in results if not r[3]]
    if failures:
        print("\nFailures:")
        for key, label, count, _, events in failures:
            reasons = [e['reason'] for e in events]
            reasons_str = f" ({', '.join(reasons)})" if reasons else ""
            print(f"  {key} ({label}): {count} event(s){reasons_str}")


if __name__ == "__main__":
    main()
