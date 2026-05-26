#!/usr/bin/env python3
import pickle
import sys

def main():
    with open("evaluation_results/detections_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    video = cache["normal/normal-11.mp4"]
    detections = video["detections"]
    
    # Let's see what was detected in the last 50 frames
    print("Last 50 frames detections:")
    for f in range(len(detections) - 50, len(detections)):
        items = [d for d in detections[f] if d['type'] in ['bottle', 'item']]
        persons = [d for d in detections[f] if d['type'] == 'person']
        print(f"Frame {f}: {len(items)} items, {len(persons)} persons")
        for i, item in enumerate(items):
            print(f"  Item {i}: bbox={item['bbox']}")

if __name__ == "__main__":
    main()
