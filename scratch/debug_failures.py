import sys
from pathlib import Path
from main import ShopliftingDetector

failing_videos = [
    "archive/normal/normal-4.mp4",
    "archive/normal/normal-6.mp4",
    "archive/normal/normal-7.mp4",
    "archive/normal/normal-8.mp4",
    "archive/normal/normal-10.mp4"
]

for video in failing_videos:
    path = Path(video)
    if not path.exists():
        print(f"{video} does not exist")
        continue
    print(f"\n--- Analyzing {video} ---")
    detector = ShopliftingDetector(time_near_shelf_threshold=15.0)
    events, _ = detector.process_video(str(path))
    for i, event in enumerate(events, 1):
        print(f"Event {i}: Person {event['track_id']}, Start: {event['start_time']:.2f}s, End: {event['end_time']:.2f}s, Reason: {event['reason']}")
