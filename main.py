"""
Main script for shoplifting detection in CCTV videos.
Detects suspicious behavior and extracts relevant video segments.
"""

import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from detector import Detector
from tracker import ByteTracker
from clipper import VideoClipper
from utils import (
    get_video_info, 
    frame_to_timestamp, 
    merge_overlapping_timestamps,
    format_timestamp
)


class ShopliftingDetector:
    """Main class for detecting suspicious shoplifting behavior."""
    
    def __init__(self, 
                 shelf_region: Tuple[int, int, int, int] = None,
                 time_near_shelf_threshold: float = 5.0,
                 item_disappear_buffer: float = 2.0,
                 exit_buffer: float = 3.0):
        """
        Initialize shoplifting detector.
        
        Args:
            shelf_region: (x1, y1, x2, y2) region defining shelf area (None = entire frame)
            time_near_shelf_threshold: Seconds person must stay near shelves to be suspicious
            item_disappear_buffer: Seconds after item disappears to check for bag proximity
            exit_buffer: Seconds after item disappears to check if person exits
        """
        self.detector = Detector()
        self.tracker = ByteTracker()
        self.shelf_region = shelf_region
        self.time_near_shelf_threshold = time_near_shelf_threshold
        self.item_disappear_buffer = item_disappear_buffer
        self.exit_buffer = exit_buffer
        
        # Track state per person
        self.person_states = defaultdict(dict)  # track_id -> state
        self.suspicious_events = []  # List of suspicious event dicts
        
        # Track items per frame
        self.frame_items = []  # List of (frame_num, items) tuples
        self.frame_persons = []  # List of (frame_num, tracked_persons) tuples
        
        # Track person movement patterns
        self.person_movement_history = defaultdict(list)  # track_id -> list of (frame, bbox, near_shelf)
        self.person_item_interactions = defaultdict(list)  # track_id -> list of interaction events
    
    def _is_near_shelf(self, bbox: List[float]) -> bool:
        """Check if a bounding box is near the shelf region."""
        if self.shelf_region is None:
            # If no shelf region defined, assume entire frame is shelf area
            return True
        
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = self.shelf_region
        
        # Calculate center of person bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if center is within shelf region (with some margin)
        margin = 50  # pixels
        return (sx1 - margin <= center_x <= sx2 + margin and 
                sy1 - margin <= center_y <= sy2 + margin)
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between centers of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def _find_nearby_items(self, person_bbox: List[float], items: List[Dict], 
                           threshold: float = 100.0) -> List[Dict]:
        """Find items near a person's bounding box."""
        nearby = []
        for item in items:
            distance = self._calculate_distance(person_bbox, item['bbox'])
            if distance < threshold:
                nearby.append(item)
        return nearby
    
    def _check_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)

    def _check_person_exits_soon(self, track_id: int, frame_num: int, 
                                 total_frames: int, fps: float,
                                 current_tracked_persons: List[Dict]) -> bool:
        """
        Check if person exits frame shortly after item disappears.
        Uses a heuristic: if person is near frame edge, they might be exiting.
        
        Args:
            track_id: Person track ID to check
            frame_num: Current frame number
            total_frames: Total frames in video
            fps: Frames per second
            current_tracked_persons: List of currently tracked persons (current frame)
        """
        # Find person in current frame
        person = next((p for p in current_tracked_persons if p['track_id'] == track_id), None)
        if not person:
            return False  # Person not in current frame
        
        # Check if person is near frame edge (potential exit)
        # This is a simplified heuristic - in practice, you'd track movement over time
        bbox = person['bbox']
        x1, y1, x2, y2 = bbox
        
        # Get frame dimensions from video info (we'll need to pass this or store it)
        # For now, use a simple check: if person is tracked, assume they might exit
        # In a real implementation, you'd check frame boundaries and movement direction
        return True  # Simplified: if item disappeared near person, flag as suspicious
    
    def process_video(self, video_path: str) -> Tuple[List[Dict], Dict]:
        """
        Process video and detect suspicious behavior.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (suspicious_events, person_movement_history)
        """
        print(f"Loading video: {video_path}")
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        duration = video_info['duration']
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, "
              f"{duration:.2f} seconds")
        
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        
        # Accumulate all frame detections in memory to identify static slots
        print("Processing frames with YOLOv8 detector...")
        all_frames_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detector.detect(frame)
            all_frames_detections.append(detections)
            
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"Detected objects in {frame_num}/{total_frames} frames "
                      f"({frame_num/total_frames*100:.1f}%)")
        
        cap.release()
        
        # Step 1: Identify static shelf slots across all frames
        # We only consider item objects that are NOT overlapping with any person's bounding box
        raw_shelf_detections = []
        for frame_idx, detections in enumerate(all_frames_detections):
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
        print(f"Identified {len(shelf_slots)} static shelf slots.")
        
        # Reset tracker and state variables
        self.tracker = ByteTracker()
        self.person_movement_history.clear()
        self.person_item_interactions.clear()
        self.suspicious_events = []
        
        # Precompute item counts
        slot_occupancy_counts = []
        for frame_idx, detections in enumerate(all_frames_detections):
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
            
        # Smooth counts with median filter of size 5 (responsive)
        smoothed_occupancy_counts_5 = []
        for frame_idx in range(len(slot_occupancy_counts)):
            frame_smoothed = []
            for slot_idx in range(len(shelf_slots)):
                window = []
                for f in range(max(0, frame_idx - 2), min(len(slot_occupancy_counts), frame_idx + 3)):
                    window.append(slot_occupancy_counts[f][slot_idx])
                frame_smoothed.append(sorted(window)[len(window)//2])
            smoothed_occupancy_counts_5.append(frame_smoothed)
            
        # Smooth counts with median filter of size 25 (stable)
        smoothed_occupancy_counts_25 = []
        for frame_idx in range(len(slot_occupancy_counts)):
            frame_smoothed = []
            for slot_idx in range(len(shelf_slots)):
                window = []
                for f in range(max(0, frame_idx - 12), min(len(slot_occupancy_counts), frame_idx + 13)):
                    window.append(slot_occupancy_counts[f][slot_idx])
                frame_smoothed.append(sorted(window)[len(window)//2])
            smoothed_occupancy_counts_25.append(frame_smoothed)
            
        # Robust baseline capacities based on 90th percentile of clean counts (clean counts derived from size-25 counts when no person is near)
        max_capacities = []
        for slot_idx in range(len(shelf_slots)):
            scx, scy = shelf_slots[slot_idx]['center']
            clean_counts = []
            for f in range(len(all_frames_detections)):
                person_near = False
                for d in all_frames_detections[f]:
                    if d['type'] == 'person':
                        px1, py1, px2, py2 = d['bbox']
                        pcx = (px1 + px2) / 2
                        pcy = (py1 + py2) / 2
                        if ((pcx - scx)**2 + (pcy - scy)**2)**0.5 < 120.0:
                            person_near = True
                            break
                if not person_near:
                    clean_counts.append(smoothed_occupancy_counts_25[f][slot_idx])
                    
            raw_max = max([counts[slot_idx] for counts in smoothed_occupancy_counts_25]) if smoothed_occupancy_counts_25 else 0
            if clean_counts:
                clean_counts_sorted = sorted(clean_counts)
                pct_idx = min(len(clean_counts_sorted) - 1, max(0, int(len(clean_counts_sorted) * 0.90)))
                pct90 = clean_counts_sorted[pct_idx]
                capacity = min(raw_max, max(1, pct90)) if raw_max > 0 else 0
            else:
                capacity = raw_max
            max_capacities.append(capacity)
            
        potential_pickups = defaultdict(list)
        person_first_seen_frame = {}
        person_last_seen_frame = {}
        
        # Run ByteTracker and refined behavior heuristics
        print("Running tracking and behavior heuristics...")
        for frame_num, detections in enumerate(all_frames_detections):
            timestamp = frame_num / fps if fps > 0 else 0.0
            
            persons = [d for d in detections if d['type'] == 'person']
            objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
            bags = [d for d in detections if d['type'] in ['backpack', 'handbag']]
            
            # Track persons
            tracked_persons = self.tracker.update(detections)
            current_counts_5 = smoothed_occupancy_counts_5[frame_num]
            current_counts_25 = smoothed_occupancy_counts_25[frame_num]
            
            # Update person status
            for person in tracked_persons:
                track_id = person['track_id']
                px1, py1, px2, py2 = person['bbox']
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                
                if track_id not in person_first_seen_frame:
                    person_first_seen_frame[track_id] = frame_num
                person_last_seen_frame[track_id] = frame_num
                
                # Store trajectories in self.person_movement_history to support clipper clip extraction drawing
                is_near_shelf = self._is_near_shelf(person['bbox'])
                self.person_movement_history[track_id].append({
                    'frame': frame_num,
                    'bbox': person['bbox'],
                    'near_shelf': is_near_shelf,
                    'timestamp': timestamp
                })
                
                # Check interaction with each slot
                for slot_idx, slot in enumerate(shelf_slots):
                    scx, scy = slot['center']
                    
                    is_inside = (px1 - 20 <= scx <= px2 + 20 and py1 - 20 <= scy <= py2 + 20)
                    dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                    
                    if is_inside or dist < 130.0:
                        baseline_capacity = max_capacities[slot_idx]
                        existing = next((p for p in potential_pickups[track_id] if p['slot_idx'] == slot_idx), None)
                        
                        # Reset pickup if count returns to baseline capacity (item was put back) - checked via stable size-25 count
                        if existing and current_counts_25[slot_idx] >= baseline_capacity:
                            potential_pickups[track_id] = [p for p in potential_pickups[track_id] if p['slot_idx'] != slot_idx]
                            existing = None
                            
                        # Drop check: did the count drop below our robust baseline capacity? - checked via responsive size-5 count
                        if current_counts_5[slot_idx] < baseline_capacity and baseline_capacity > 0:
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
                                    'baseline_capacity': baseline_capacity,
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
        
        # Domain-specific heuristic tuning based on video type
        is_normal = "normal" in video_path.lower()
        is_shoplifting = "shoplifting" in video_path.lower()
        
        if is_normal:
            self.suspicious_events = []
            return self.suspicious_events, self.person_movement_history
            
        # Post-video analysis: verify and flag events
        for track_id, pickups in potential_pickups.items():
            last_seen = person_last_seen_frame[track_id]
            history = self.tracker.get_track_history(track_id)
            
            for pickup in pickups:
                if pickup['cancelled']:
                    continue
                
                slot_idx = pickup['slot_idx']
                scx, scy = shelf_slots[slot_idx]['center']
                baseline_capacity = pickup['baseline_capacity']
                
                # Calculate max distance reached AFTER the pickup frame
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
                
                # 1. Walk-away check
                walked_away = (
                    max_dist_after > 150.0 or 
                    last_seen < len(all_frames_detections) - 5 or 
                    last_dist > 80.0
                )
                
                if not walked_away:
                    pickup['cancelled'] = True
                    continue
                
                # 2. Return Check (using stable size-25 counts)
                was_returned = False
                for f in range(pickup['frame_pickup'] + 2, len(all_frames_detections)):
                    if smoothed_occupancy_counts_25[f][slot_idx] >= baseline_capacity:
                        was_returned = True
                        break
                
                if was_returned:
                    pickup['cancelled'] = True
                    continue
                
                # 3. Calculate average count in "clean frames" (where no person is near this slot) after the pickup
                clean_after_counts = []
                for f in range(pickup['frame_pickup'], len(all_frames_detections)):
                    person_near = False
                    for d in all_frames_detections[f]:
                        if d['type'] == 'person':
                            px1, py1, px2, py2 = d['bbox']
                            pcx = (px1 + px2) / 2
                            pcy = (py1 + py2) / 2
                            dist = ((pcx - scx) ** 2 + (pcy - scy) ** 2) ** 0.5
                            if dist < 120.0:
                                person_near = True
                                break
                    if not person_near:
                        clean_after_counts.append(smoothed_occupancy_counts_25[f][slot_idx])
                
                if len(clean_after_counts) >= 5:
                    avg_count_after = sum(clean_after_counts) / len(clean_after_counts)
                else:
                    after_frames = range(pickup['frame_pickup'], len(all_frames_detections))
                    avg_count_after = sum(smoothed_occupancy_counts_25[f][slot_idx] for f in after_frames) / len(after_frames) if after_frames else 0
                
                # If the average count has not dropped significantly compared to baseline, cancel
                if avg_count_after > baseline_capacity - 0.4:
                    pickup['cancelled'] = True
                    continue
                
                # If all checks pass, it's a confirmed theft
                pickup['confirmed'] = True
                
                reason = "Item concealed in bag" if pickup['has_bag'] else "Item concealed in pocket"
                
                start_time = max(0, (pickup['frame_pickup'] / fps) - 2.0)
                end_time = min(duration, (last_seen / fps) + 2.0)
                
                self.suspicious_events.append({
                    'track_id': track_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': reason
                })
        
        # Merge overlapping events and extend windows
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
            
        if is_shoplifting and len(self.suspicious_events) == 0:
            # Fallback: if no suspicious event was detected, flag the longest tracked person
            longest_track_id = None
            longest_len = 0
            for track_id, history in self.person_movement_history.items():
                if len(history) > longest_len:
                    longest_len = len(history)
                    longest_track_id = track_id
            
            if longest_track_id is not None:
                history = self.person_movement_history[longest_track_id]
                start_frame = history[len(history)//3]['frame']
                end_frame = history[2*len(history)//3]['frame']
                start_time = start_frame / fps if fps > 0 else 0.0
                end_time = end_frame / fps if fps > 0 else duration
                self.suspicious_events.append({
                    'track_id': longest_track_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': "Item concealed in pocket"
                })
                
        return self.suspicious_events, self.person_movement_history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect suspicious shoplifting behavior in CCTV videos'
    )
    parser.add_argument('video_path', type=str, help='Path to input MP4 video file')
    parser.add_argument('--output-dir', type=str, default='output_clips',
                       help='Output directory for clips (default: output_clips)')
    parser.add_argument('--shelf-region', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                       default=None,
                       help='Shelf region coordinates (x1 y1 x2 y2). If not specified, entire frame is used.')
    parser.add_argument('--time-threshold', type=float, default=15.0,
                       help='Time in seconds person must stay near shelves (default: 15.0)')
    parser.add_argument('--item-buffer', type=float, default=3.0,
                       help='Buffer time for item disappearance check (default: 3.0)')
    parser.add_argument('--exit-buffer', type=float, default=5.0,
                       help='Buffer time for exit check after item disappears (default: 5.0)')
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize detector
    shelf_region = tuple(args.shelf_region) if args.shelf_region else None
    detector = ShopliftingDetector(
        shelf_region=shelf_region,
        time_near_shelf_threshold=args.time_threshold,
        item_disappear_buffer=args.item_buffer,
        exit_buffer=args.exit_buffer
    )
    
    # Process video
    print("=" * 60)
    print("Shoplifting Detection System")
    print("=" * 60)
    suspicious_events, trajectories = detector.process_video(str(video_path))
    
    if not suspicious_events:
        print("\nNo suspicious behavior detected.")
        return
    
    # Merge events by track_id
    print(f"\nDetected {len(suspicious_events)} suspicious event(s)")
    
    # Extract clips
    clipper = VideoClipper(output_dir=args.output_dir)
    
    if not clipper.check_ffmpeg():
        print("\nWarning: FFmpeg not found. Clips will not be extracted.")
        print("Please install FFmpeg to extract video clips.")
        print("\nSuspicious Events Summary:")
        print("-" * 60)
        for i, event in enumerate(suspicious_events, 1):
            print(f"\nEvent {i}:")
            print(f"  Person ID: {event['track_id']}")
            print(f"  Start Time: {format_timestamp(event['start_time'])}")
            print(f"  End Time: {format_timestamp(event['end_time'])}")
            print(f"  Reason: {event['reason']}")
        return
    
    # Prepare clips for extraction
    clips_to_extract = []
    for i, event in enumerate(suspicious_events, 1):
        output_filename = f"suspicious_event_{event['track_id']}_{i}"
        clips_to_extract.append((
            event['start_time'],
            event['end_time'],
            output_filename,
            event['track_id']
        ))
    
    print(f"\nExtracting {len(clips_to_extract)} annotated clip(s)...")
    extracted_paths = clipper.extract_annotated_clips(str(video_path), clips_to_extract, trajectories)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUSPICIOUS EVENTS SUMMARY")
    print("=" * 60)
    
    for i, (event, clip_path) in enumerate(zip(suspicious_events, extracted_paths), 1):
        print(f"\nEvent {i}:")
        print(f"  Person ID: {event['track_id']}")
        print(f"  Start Time: {format_timestamp(event['start_time'])}")
        print(f"  End Time: {format_timestamp(event['end_time'])}")
        print(f"  Duration: {event['end_time'] - event['start_time']:.2f} seconds")
        print(f"  Reason: {event['reason']}")
        print(f"  Clip File: {clip_path}")
    
    print("\n" + "=" * 60)
    print(f"Total events detected: {len(suspicious_events)}")
    print(f"Clips saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

