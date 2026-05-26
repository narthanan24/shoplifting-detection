#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scratch.run_full_simulation import process_video

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    print("Analyzing normal video failures...")
    for key, video_data in sorted(cache.items()):
        if not key.startswith("normal/"):
            continue
            
        events = process_video(video_data)
        if len(events) > 0:
            print(f"FAIL {key}: {len(events)} events detected:")
            for e in events:
                print(f"  Person {e['track_id']}, time {e['start_time']:.1f}s - {e['end_time']:.1f}s, reason: {e['reason']}")

if __name__ == "__main__":
    main()
