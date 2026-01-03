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
    
    def _check_item_taken_from_shelf(self, track_id: int, frame_num: int, 
                                     fps: float, video_info: Dict) -> Tuple[bool, float]:
        """
        Check if an item was taken from shelf and person is leaving (not returning).
        Distinguishes between returning items vs taking and leaving.
        Returns (is_theft, timestamp) if detected.
        """
        lookback_frames = int(self.item_disappear_buffer * fps * 4)
        start_frame = max(0, frame_num - lookback_frames)
        
        # Track shelf items over time and person position
        shelf_item_counts = []  # List of (frame_num, item_count) tuples
        person_positions = []  # List of (frame_num, near_shelf) tuples
        
        # Count items on shelves and track person position
        for f_item_idx in range(len(self.frame_items)):
            frame_num_items, items = self.frame_items[f_item_idx]
            if start_frame <= frame_num_items <= frame_num:
                shelf_items = [item for item in items if self._is_near_shelf(item['bbox'])]
                shelf_item_counts.append((frame_num_items, len(shelf_items)))
        
        # Track person's position relative to shelf
        for f_idx in range(len(self.frame_persons)):
            frame_num_hist, persons = self.frame_persons[f_idx]
            if start_frame <= frame_num_hist <= frame_num:
                person = next((p for p in persons if p['track_id'] == track_id), None)
                if person:
                    is_near = self._is_near_shelf(person['bbox'])
                    person_positions.append((frame_num_hist, is_near))
        
        if len(shelf_item_counts) < 10 or len(person_positions) < 10:
            return False, 0.0  # Not enough history
        
        # Look for pattern: item count decreases, person was near, then person leaves and doesn't return
        # This distinguishes theft (take and leave) from returning (take, then come back)
        
        # Find when item count decreased
        for i in range(1, len(shelf_item_counts)):
            prev_frame, prev_count = shelf_item_counts[i-1]
            curr_frame, curr_count = shelf_item_counts[i]
            
            if prev_count > curr_count:  # Item disappeared
                # Check if person was near shelf when item disappeared
                person_was_near_when_taken = False
                for pos_frame, is_near in person_positions:
                    if abs(pos_frame - curr_frame) <= 10:  # Within 10 frames
                        if is_near:
                            person_was_near_when_taken = True
                            break
                
                if person_was_near_when_taken:
                    # Check if person leaves and doesn't return (theft pattern)
                    # vs returns to shelf (returning item pattern)
                    frames_after = 60  # Check next 2 seconds (60 frames at 30fps)
                    person_returned = False
                    person_left = False
                    
                    for pos_frame, is_near in person_positions:
                        if curr_frame < pos_frame <= curr_frame + frames_after:
                            if not is_near:
                                person_left = True
                            elif is_near and person_left:
                                # Person left then came back - likely returning item, not theft
                                person_returned = True
                                break
                    
                    # If person left and didn't return within reasonable time, it's theft
                    if person_left and not person_returned:
                        # Additional check: look ahead to see if person stays away
                        lookahead_frames = 90  # 3 seconds
                        stayed_away = True
                        for pos_frame, is_near in person_positions:
                            if curr_frame < pos_frame <= curr_frame + lookahead_frames:
                                if is_near:
                                    stayed_away = False
                                    break
                        
                        if stayed_away:
                            timestamp = frame_to_timestamp(curr_frame, fps)
                            return True, timestamp
        
        return False, 0.0
    
    def _analyze_full_timeline_for_theft(self, fps: float, video_info: Dict) -> List[Dict]:
        """
        Post-process the entire video timeline to find theft patterns.
        Looks for: item count decreases, person was near, person leaves and doesn't return.
        """
        theft_events = []
        
        # Analyze item counts over time per track
        track_ids = set()
        for _, persons in self.frame_persons:
            for person in persons:
                track_ids.add(person['track_id'])
        
        for track_id in track_ids:
            # Build timeline of item counts and person positions
            timeline = []  # List of (frame_num, item_count, person_near_shelf)
            
            for f_item_idx in range(len(self.frame_items)):
                frame_num_items, items = self.frame_items[f_item_idx]
                shelf_items = [item for item in items if self._is_near_shelf(item['bbox'])]
                item_count = len(shelf_items)
                
                # Find if person was near shelf at this frame
                person_near = False
                for f_idx in range(len(self.frame_persons)):
                    frame_num_persons, persons = self.frame_persons[f_idx]
                    if frame_num_persons == frame_num_items:
                        person = next((p for p in persons if p['track_id'] == track_id), None)
                        if person and self._is_near_shelf(person['bbox']):
                            person_near = True
                            break
                
                timeline.append((frame_num_items, item_count, person_near))
            
            if len(timeline) < 20:
                continue
            
            # Look for pattern: item count drops, person was near, then person leaves
            for i in range(10, len(timeline) - 30):  # Need enough history and future
                prev_frame, prev_count, prev_near = timeline[i-1]
                curr_frame, curr_count, curr_near = timeline[i]
                
                # Item count decreased and person was near
                if prev_count > curr_count and prev_near:
                    # Check if person leaves and doesn't return
                    person_left = False
                    person_returned = False
                    
                    for j in range(i+1, min(i+90, len(timeline))):  # Check next 3 seconds
                        _, _, near = timeline[j]
                        if not near:
                            person_left = True
                        elif near and person_left:
                            person_returned = True
                            break
                    
                    # If person left and didn't return, it's theft
                    if person_left and not person_returned:
                        start_time = max(0, frame_to_timestamp(curr_frame, fps) - 3.0)
                        end_time = min(video_info['duration'], frame_to_timestamp(curr_frame, fps) + 7.0)
                        
                        theft_events.append({
                            'track_id': track_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'reason': 'Item taken from shelf and person left (theft detected)'
                        })
                        break  # Only one theft event per person
        
        return theft_events
    
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
    
    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process video and detect suspicious behavior.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            List of suspicious events, each with:
            - 'track_id': Person track ID
            - 'start_time': Start timestamp in seconds
            - 'end_time': End timestamp in seconds
            - 'reason': Reason for flagging
        """
        print(f"Loading video: {video_path}")
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, "
              f"{video_info['duration']:.2f} seconds")
        
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        
        # Initialize person states
        person_near_shelf_time = defaultdict(float)  # track_id -> seconds near shelf
        person_first_seen = {}  # track_id -> first frame number
        person_last_seen = {}  # track_id -> last frame number
        
        print("Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_to_timestamp(frame_num, fps)
            
            # Detect objects
            detections = self.detector.detect(frame)
            persons = self.detector.get_persons(detections)
            items = self.detector.get_items(detections)
            
            # Track persons
            tracked_persons = self.tracker.update(detections)
            
            # Store frame data for analysis
            self.frame_items.append((frame_num, items))
            self.frame_persons.append((frame_num, tracked_persons))
            
            # Update person states
            active_track_ids = set()
            for person in tracked_persons:
                track_id = person['track_id']
                active_track_ids.add(track_id)
                
                # Initialize person state
                if track_id not in person_first_seen:
                    person_first_seen[track_id] = frame_num
                    person_near_shelf_time[track_id] = 0.0
                
                person_last_seen[track_id] = frame_num
                
                # Track movement history
                is_near_shelf = self._is_near_shelf(person['bbox'])
                self.person_movement_history[track_id].append({
                    'frame': frame_num,
                    'bbox': person['bbox'],
                    'near_shelf': is_near_shelf,
                    'timestamp': timestamp
                })
                
                # Check if person is near shelf
                if is_near_shelf:
                    person_near_shelf_time[track_id] += (1.0 / fps)
                    
                    # Check for items near person while at shelf
                    nearby_items = self._find_nearby_items(person['bbox'], items, threshold=200.0)
                    if nearby_items:
                        self.person_item_interactions[track_id].append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'items_count': len(nearby_items),
                            'near_shelf': True
                        })
                else:
                    # Person moved away from shelf - check if they were there before
                    if person_near_shelf_time[track_id] > 1.0:  # Was near shelf for at least 1 second
                        # Check if items decreased when person left
                        prev_items = 0
                        current_items_near = len(self._find_nearby_items(person['bbox'], items, threshold=200.0))
                        
                        # Look at recent interactions
                        if self.person_item_interactions[track_id]:
                            prev_items = self.person_item_interactions[track_id][-1].get('items_count', 0)
                        
                        # If person was near items and now moved away, it's suspicious
                        if prev_items > 0:
                            self.person_item_interactions[track_id].append({
                                'frame': frame_num,
                                'timestamp': timestamp,
                                'items_count': current_items_near,
                                'near_shelf': False,
                                'moved_away_after_interaction': True
                            })
                    
                    # Reset time if person moves away (but keep history)
                    if person_near_shelf_time[track_id] < self.time_near_shelf_threshold:
                        person_near_shelf_time[track_id] = 0.0
            
            # Check for suspicious behavior
            for person in tracked_persons:
                track_id = person['track_id']
                
                # Heuristic 1: Person stays near shelves too long (less sensitive, only if no item interaction)
                # Skip this if we're detecting actual theft patterns (Heuristic 2)
                if person_near_shelf_time[track_id] >= self.time_near_shelf_threshold * 2:  # Double threshold
                    # Only flag if person hasn't had clear item interactions (to avoid false positives on returns)
                    has_item_interaction = any(
                        interaction.get('items_count', 0) > 0 
                        for interaction in self.person_item_interactions[track_id]
                    )
                    
                    if not has_item_interaction:
                        # Extend window to include approach and departure
                        start_time = max(0, frame_to_timestamp(person_first_seen[track_id], fps) - 2.0)
                        end_time = min(video_info['duration'], timestamp + 5.0)  # Include walk-away
                        
                        # Check if we already have an event for this person
                        existing = next((e for e in self.suspicious_events 
                                       if e['track_id'] == track_id and 
                                       abs(e['start_time'] - start_time) < 3.0), None)
                        
                        if not existing:
                            self.suspicious_events.append({
                                'track_id': track_id,
                                'start_time': start_time,
                                'end_time': end_time,
                                'reason': f'Stayed near shelves for {person_near_shelf_time[track_id]:.1f}s'
                            })
                
                # Heuristic 2: Item taken from shelf and person leaves (actual theft) - PRIORITY
                is_theft, theft_timestamp = self._check_item_taken_from_shelf(
                    track_id, frame_num, fps, video_info
                )
                if is_theft:
                    # Calculate event time window around the theft
                    start_time = max(0, theft_timestamp - 3.0)  # 3 seconds before to show approach
                    end_time = min(video_info['duration'], timestamp + 6.0)  # Include full walk-away
                    
                    # Remove any existing events for this person that might be false positives
                    self.suspicious_events = [
                        e for e in self.suspicious_events 
                        if not (e['track_id'] == track_id and 
                               'Stayed near shelves' in e['reason'])
                    ]
                    
                    existing = next((e for e in self.suspicious_events 
                                   if e['track_id'] == track_id and 
                                   abs(e['start_time'] - start_time) < 2.0), None)
                    
                    if not existing:
                        self.suspicious_events.append({
                            'track_id': track_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'reason': 'Item taken from shelf and person left (theft detected)'
                        })
                
                # Heuristic 3: Person near shelf and then moves away quickly (grab and go)
                if person_near_shelf_time[track_id] > 1.5:  # Was near shelf for at least 1.5 seconds
                    # Check if person is moving away from shelf
                    if not self._is_near_shelf(person['bbox']):
                        # Check if person had item interactions
                        had_item_interaction = any(
                            interaction.get('items_count', 0) > 0 
                            for interaction in self.person_item_interactions[track_id]
                            if interaction.get('near_shelf', False)
                        )
                        
                        # Person was near shelf but now moved away - could be suspicious
                        start_time = max(0, timestamp - 4.0)  # Include more context before
                        end_time = min(video_info['duration'], timestamp + 3.0)  # Include walk-away
                        
                        existing = next((e for e in self.suspicious_events 
                                       if e['track_id'] == track_id and 
                                       abs(e['start_time'] - start_time) < 3.0), None)
                        
                        if not existing:
                            reason = f'Person was near shelf ({person_near_shelf_time[track_id]:.1f}s) then moved away'
                            if had_item_interaction:
                                reason += ' (after item interaction - possible theft)'
                            
                            self.suspicious_events.append({
                                'track_id': track_id,
                                'start_time': start_time,
                                'end_time': end_time,
                                'reason': reason
                            })
                
                # Heuristic 4: Detect item disappearance pattern (item was near person, then gone)
                # This catches the concealment action
                if len(self.person_item_interactions[track_id]) >= 2:
                    recent_interactions = self.person_item_interactions[track_id][-2:]
                    prev_interaction = recent_interactions[0]
                    curr_interaction = recent_interactions[1]
                    
                    # If person had items near them at shelf, then items decreased/disappeared
                    if (prev_interaction.get('near_shelf', False) and 
                        prev_interaction.get('items_count', 0) > 0 and
                        curr_interaction.get('items_count', 0) < prev_interaction.get('items_count', 0)):
                        
                        start_time = max(0, prev_interaction.get('timestamp', timestamp) - 2.0)
                        end_time = min(video_info['duration'], curr_interaction.get('timestamp', timestamp) + 4.0)
                        
                        existing = next((e for e in self.suspicious_events 
                                       if e['track_id'] == track_id and 
                                       abs(e['start_time'] - start_time) < 2.0), None)
                        
                        if not existing:
                            self.suspicious_events.append({
                                'track_id': track_id,
                                'start_time': start_time,
                                'end_time': end_time,
                                'reason': 'Item disappeared near person (possible concealment/theft)'
                            })
            
            frame_num += 1
            
            if frame_num % 30 == 0:
                print(f"Processed {frame_num}/{total_frames} frames "
                      f"({frame_num/total_frames*100:.1f}%)")
        
        cap.release()
        
        # Post-process: Analyze full timeline for theft patterns
        print("\nAnalyzing full video timeline for theft patterns...")
        theft_events = self._analyze_full_timeline_for_theft(fps, video_info)
        
        # Prioritize theft events - remove false positives and keep only theft events
        if theft_events:
            # Remove non-theft events for the same person
            for theft_event in theft_events:
                self.suspicious_events = [
                    e for e in self.suspicious_events 
                    if not (e['track_id'] == theft_event['track_id'] and 
                           'Stayed near shelves' in e['reason'] and
                           'theft' not in e['reason'].lower())
                ]
                self.suspicious_events.append(theft_event)
        
        # Merge overlapping events and extend windows
        if self.suspicious_events:
            # Prioritize theft events - don't extend them as much, use their specific timestamps
            extended_events = []
            for event in self.suspicious_events:
                if 'theft detected' in event['reason'].lower():
                    # Theft events: use their specific timestamps (already calculated)
                    extended_events.append(event)
                else:
                    # Other events: extend window
                    extended_start = max(0, event['start_time'] - 2.0)
                    extended_end = min(video_info['duration'], event['end_time'] + 4.0)
                    extended_events.append({
                        'track_id': event['track_id'],
                        'start_time': extended_start,
                        'end_time': extended_end,
                        'reason': event['reason']
                    })
            
            # Only merge if events are very close together (within 1 second)
            intervals = [(e['start_time'], e['end_time']) for e in extended_events]
            merged_intervals = merge_overlapping_timestamps(intervals, merge_threshold=1.0)
            
            # Update events with merged intervals
            merged_events = []
            for start, end in merged_intervals:
                # Find original events that overlap with this interval
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
        
        return self.suspicious_events


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
    parser.add_argument('--time-threshold', type=float, default=5.0,
                       help='Time in seconds person must stay near shelves (default: 5.0)')
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
    suspicious_events = detector.process_video(str(video_path))
    
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
            output_filename
        ))
    
    print(f"\nExtracting {len(clips_to_extract)} clip(s)...")
    extracted_paths = clipper.extract_clips(str(video_path), clips_to_extract)
    
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

