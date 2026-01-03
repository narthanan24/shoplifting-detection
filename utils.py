"""
Utility functions for video processing and timestamp management.
"""

import cv2
from typing import List, Tuple, Dict


def get_video_info(video_path: str) -> Dict:
    """
    Extract video information (FPS, frame count, duration).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with 'fps', 'frame_count', 'duration', 'width', 'height'
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }


def frame_to_timestamp(frame_number: int, fps: float) -> float:
    """
    Convert frame number to timestamp in seconds.
    
    Args:
        frame_number: Frame number (0-indexed)
        fps: Frames per second
        
    Returns:
        Timestamp in seconds
    """
    return frame_number / fps if fps > 0 else 0.0


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """
    Convert timestamp to frame number.
    
    Args:
        timestamp: Timestamp in seconds
        fps: Frames per second
        
    Returns:
        Frame number (0-indexed)
    """
    return int(timestamp * fps)


def merge_overlapping_timestamps(intervals: List[Tuple[float, float]], 
                                  merge_threshold: float = 2.0) -> List[Tuple[float, float]]:
    """
    Merge overlapping or nearby timestamp intervals.
    
    Args:
        intervals: List of (start, end) timestamp tuples
        merge_threshold: Maximum gap in seconds between intervals to merge
        
    Returns:
        List of merged (start, end) timestamp tuples
    """
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        
        # If current interval overlaps or is close to last, merge them
        if current[0] <= last[1] + merge_threshold:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp in seconds to HH:MM:SS.mmm format.
    
    Args:
        seconds: Timestamp in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

