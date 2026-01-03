"""
Person tracking module using ByteTrack algorithm.
Tracks persons across frames and maintains their state.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ByteTracker:
    """
    Simplified ByteTrack implementation for person tracking.
    Tracks persons across frames and maintains their history.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep a track without detection
            min_hits: Minimum detections before a track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.tracks = {}  # track_id -> track info
        self.next_id = 1
        self.track_history = defaultdict(list)  # track_id -> list of frame states
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of person detections with 'bbox' key
            
        Returns:
            List of tracked persons with 'track_id' added
        """
        self.frame_count += 1
        
        # Filter to only person detections
        person_detections = [d for d in detections if d.get('type') == 'person']
        
        if len(person_detections) == 0:
            # No detections, age all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # First pass: match high confidence detections to confirmed tracks
        for track_id, track in self.tracks.items():
            if track['age'] > 0:
                continue  # Skip tracks that were lost
            
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(person_detections):
                if det_idx in matched_detections:
                    continue
                
                iou = self._iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Match found
                det = person_detections[best_det_idx]
                track['bbox'] = det['bbox']
                track['confidence'] = det['confidence']
                track['age'] = 0
                track['hits'] += 1
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
        
        # Second pass: match remaining detections to unconfirmed tracks
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks or track['hits'] >= self.min_hits:
                continue
            
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(person_detections):
                if det_idx in matched_detections:
                    continue
                
                iou = self._iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                det = person_detections[best_det_idx]
                track['bbox'] = det['bbox']
                track['confidence'] = det['confidence']
                track['age'] = 0
                track['hits'] += 1
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
        
        # Age unmatched tracks
        for track_id in list(self.tracks.keys()):  # Create list to avoid modification during iteration
            if track_id not in matched_tracks:
                track = self.tracks[track_id]
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(person_detections):
            if det_idx not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                
                center = self._calculate_center(det['bbox'])
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'hits': 1,
                    'center': center
                }
                matched_tracks.add(track_id)
        
        # Build output with track IDs
        tracked_detections = []
        for track_id in list(self.tracks.keys()):  # Create list to avoid modification during iteration
            track = self.tracks[track_id]
            if track['age'] == 0:  # Only return active tracks
                center = self._calculate_center(track['bbox'])
                track['center'] = center
                
                # Store history
                self.track_history[track_id].append({
                    'frame': self.frame_count,
                    'bbox': track['bbox'],
                    'center': center
                })
                
                tracked_detections.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'center': center,
                    'hits': track['hits'],
                    'type': 'person'
                })
        
        return tracked_detections
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get history for a specific track."""
        return self.track_history.get(track_id, [])
    
    def get_track_duration(self, track_id: int, fps: float) -> float:
        """Get duration in seconds that a track has been active."""
        history = self.track_history.get(track_id, [])
        if len(history) == 0:
            return 0.0
        return len(history) / fps

