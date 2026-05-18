import re

with open("main.py", "r") as f:
    content = f.read()

# 1. Update return type hint of process_video
content = content.replace(
    "def process_video(self, video_path: str) -> List[Dict]:",
    "def process_video(self, video_path: str) -> Tuple[List[Dict], Dict]:"
)

# 2. Update return statement of process_video
content = content.replace(
    "return self.suspicious_events",
    "return self.suspicious_events, self.person_movement_history"
)

# 3. Update main() to capture trajectories
content = content.replace(
    "suspicious_events = detector.process_video(str(video_path))",
    "suspicious_events, trajectories = detector.process_video(str(video_path))"
)

# 4. Update clips_to_extract to include track_id
old_clips_append = """        clips_to_extract.append((
            event['start_time'],
            event['end_time'],
            output_filename
        ))"""

new_clips_append = """        clips_to_extract.append((
            event['start_time'],
            event['end_time'],
            output_filename,
            event['track_id']
        ))"""

content = content.replace(old_clips_append, new_clips_append)

# 5. Update extraction call
old_extract_call = """    print(f"\\nExtracting {len(clips_to_extract)} clip(s)...")
    extracted_paths = clipper.extract_clips(str(video_path), clips_to_extract)"""

new_extract_call = """    print(f"\\nExtracting {len(clips_to_extract)} annotated clip(s)...")
    extracted_paths = clipper.extract_annotated_clips(str(video_path), clips_to_extract, trajectories)"""

content = content.replace(old_extract_call, new_extract_call)

with open("main.py", "w") as f:
    f.write(content)
