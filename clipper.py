"""
Video clipping module using FFmpeg to extract suspicious segments.
"""

import subprocess
import cv2
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path


class VideoClipper:
    """Extract video clips using FFmpeg."""
    
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize video clipper.
        
        Args:
            output_dir: Directory to save output clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def extract_clip(self, video_path: str, start_time: float, end_time: float, 
                    output_filename: str) -> str:
        """
        Extract a video clip using FFmpeg.
        
        Args:
            video_path: Path to input video
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            output_filename: Output filename (without extension)
            
        Returns:
            Path to extracted clip file
        """
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed. Please install FFmpeg to extract clips.")
        
        duration = end_time - start_time
        output_path = self.output_dir / f"{output_filename}.mp4"
        
        # FFmpeg command to extract clip
        # -ss: start time, -t: duration, -c copy: copy codec (faster, no re-encoding)
        # -avoid_negative_ts make_zero: handle timestamp issues
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, 
                          capture_output=True, 
                          check=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            # If copy fails, try re-encoding
            cmd_reencode = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                str(output_path)
            ]
            try:
                subprocess.run(cmd_reencode,
                             capture_output=True,
                             check=True)
                return str(output_path)
            except subprocess.CalledProcessError as e2:
                error_msg = e2.stderr.decode() if e2.stderr else str(e2)
                raise RuntimeError(f"Failed to extract clip: {error_msg}")
    
    def extract_clips(self, video_path: str, 
                     clips: List[Tuple[float, float, str]]) -> List[str]:
        """
        Extract multiple video clips.
        
        Args:
            video_path: Path to input video
            clips: List of (start_time, end_time, output_filename) tuples
            
        Returns:
            List of paths to extracted clip files
        """
        output_paths = []
        for start_time, end_time, output_filename in clips:
            try:
                path = self.extract_clip(video_path, start_time, end_time, output_filename)
                output_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to extract clip {output_filename}: {e}")
        
        return output_paths


    def extract_annotated_clip(self, video_path: str, start_time: float, end_time: float, 
                              output_filename: str, track_id: int, 
                              person_history: List[Dict[str, Any]]) -> str:
        """
        Extract a video clip and draw a red rectangle around the suspicious person.
        """
        output_path = self.output_dir / f"{output_filename}.mp4"
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Build a lookup for this person's bounding box by frame number
        bbox_by_frame = {h['frame']: h['bbox'] for h in person_history}
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if we have a bounding box for this person in this frame
            if current_frame in bbox_by_frame:
                x1, y1, x2, y2 = bbox_by_frame[current_frame]
                
                # Draw thick red rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                
                # Add text label
                cv2.putText(frame, f"Suspicious Activity (ID: {track_id})", 
                           (int(x1), max(10, int(y1) - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(frame)
            current_frame += 1
            
        cap.release()
        out.release()
        
        return str(output_path)

    def extract_annotated_clips(self, video_path: str, 
                               clips: List[Tuple[float, float, str, int]], 
                               trajectories: Dict[int, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract multiple annotated video clips.
        
        Args:
            video_path: Path to input video
            clips: List of (start_time, end_time, output_filename, track_id) tuples
            trajectories: Dictionary mapping track_id to person_movement_history
            
        Returns:
            List of paths to extracted clip files
        """
        output_paths = []
        for start_time, end_time, output_filename, track_id in clips:
            try:
                person_history = trajectories.get(track_id, [])
                path = self.extract_annotated_clip(video_path, start_time, end_time, 
                                                 output_filename, track_id, person_history)
                output_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to extract annotated clip {output_filename}: {e}")
        
        return output_paths
