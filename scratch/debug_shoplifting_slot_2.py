import pickle
from fast_eval import SimulatedDetector

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)

video_data = cache['shoplifting/shoplifting-1.mp4']
frames_detections = video_data['detections']
fps = video_data['metadata']['fps']

# Let's inspect Slot 2 center from debug: (340.0, 391.5)
scx, scy = 340.0, 391.5

print("Occupancy log of Slot 2 (340.0, 391.5) in shoplifting-1.mp4:")
for frame_num, detections in enumerate(frames_detections):
    objects = [d for d in detections if d['type'] not in ['backpack', 'handbag', 'person']]
    occupied = False
    occupying_obj = None
    for obj in objects:
        ox1, oy1, ox2, oy2 = obj['bbox']
        ocx = (ox1 + ox2) / 2
        ocy = (oy1 + oy2) / 2
        dist = ((ocx - scx) ** 2 + (ocy - scy) ** 2) ** 0.5
        if dist < 40.0:
            occupied = True
            occupying_obj = obj
            break
            
    if occupied:
        print(f"  Frame {frame_num} ({frame_num/fps:.2f}s): Occupied by {occupying_obj['class_name']} conf={occupying_obj['confidence']:.2f} bbox={occupying_obj['bbox']}")
