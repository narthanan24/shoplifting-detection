import pickle
from pathlib import Path

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)

video_data = cache['shoplifting/shoplifting-1.mp4']
frames = video_data['detections']
fps = video_data['metadata']['fps']

# Inspect frames 30 to 50
print("Frames 30 to 50 in shoplifting-1.mp4:")
for f_idx in range(30, 50):
    detections = frames[f_idx]
    persons = [d for d in detections if d['type'] == 'person']
    objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
    
    print(f"\nFrame {f_idx} ({f_idx/fps:.2f}s):")
    for p in persons:
        print(f"  Person {p['class_id']}: bbox={p['bbox']}")
    for obj in objects:
        print(f"  Object {obj['class_name']} ({obj['class_id']}): bbox={obj['bbox']} conf={obj['confidence']:.2f}")
