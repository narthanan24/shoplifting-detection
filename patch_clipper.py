import re

with open("clipper.py", "r") as f:
    content = f.read()

# Add cv2 import if missing
if 'import cv2' not in content:
    content = content.replace("import subprocess", "import subprocess\nimport cv2")
if 'from typing import List, Tuple, Dict' not in content:
    content = content.replace("from typing import List, Tuple", "from typing import List, Tuple, Dict, Any")

annotated_methods = """
    def extract_annotated_clip(self, video_path: str, start_time: float, end_time: float, 
                              output_filename: str, track_id: int, 
                              person_history: List[Dict[str, Any]]) -> str:
        \"\"\"
        Extract a video clip and draw a red rectangle around the suspicious person.
        \"\"\"
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
        \"\"\"
        Extract multiple annotated video clips.
        
        Args:
            video_path: Path to input video
            clips: List of (start_time, end_time, output_filename, track_id) tuples
            trajectories: Dictionary mapping track_id to person_movement_history
            
        Returns:
            List of paths to extracted clip files
        \"\"\"
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
"""

# Append to the end of the VideoClipper class
content += annotated_methods

with open("clipper.py", "w") as f:
    f.write(content)
