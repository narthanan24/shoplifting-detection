"""
Video clipping module using FFmpeg to extract suspicious segments.
"""

import subprocess
import os
from typing import List, Tuple
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

