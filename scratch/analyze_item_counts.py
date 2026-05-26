import pickle
from pathlib import Path

cache_path = Path("evaluation_results/detections_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

for key, video_data in cache.items():
    frames = video_data['detections']
    counts = [len([d for d in f if d['type'] in ['backpack', 'handbag', 'bottle', 'item']]) for f in frames]
    
    # Calculate start avg (first 30 frames) and end avg (last 30 frames)
    start_avg = sum(counts[:30]) / min(30, len(counts))
    end_avg = sum(counts[-30:]) / min(30, len(counts))
    
    print(f"{key}: start={start_avg:.2f}, end={end_avg:.2f}, diff={end_avg - start_avg:.2f}")
