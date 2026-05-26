#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path

def explore(video_key, cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if video_key not in cache:
        print(f"{video_key} not in cache.")
        return
    
    video_data = cache[video_key]
    frames_detections = video_data['detections']
    print(f"Total frames: {len(frames_detections)}")
    for f_idx, detections in enumerate(frames_detections):
        persons = [d for d in detections if d['type'] == 'person']
        if persons:
            print(f"Frame {f_idx}: {len(persons)} person(s)")
            for idx, p in enumerate(persons):
                print(f"  Person {idx}: bbox={p['bbox']}, conf={p['confidence']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: explore_detections.py <video_key>")
        sys.exit(1)
    explore(sys.argv[1], "evaluation_results/detections_cache.pkl")
