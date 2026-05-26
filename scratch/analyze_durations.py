#!/usr/bin/env python3
import pickle
from pathlib import Path

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    normals = [v for k, v in cache.items() if k.startswith("normal/")]
    shops = [v for k, v in cache.items() if k.startswith("shoplifting/")]
    
    print("Normal videos durations:")
    n_durations = [v['metadata']['duration'] for v in normals]
    print(f"  Count: {len(n_durations)}")
    if n_durations:
        print(f"  Min: {min(n_durations):.2f}s, Max: {max(n_durations):.2f}s, Avg: {sum(n_durations)/len(n_durations):.2f}s")
        
    print("\nShoplifting videos durations:")
    s_durations = [v['metadata']['duration'] for v in shops]
    print(f"  Count: {len(s_durations)}")
    if s_durations:
        print(f"  Min: {min(s_durations):.2f}s, Max: {max(s_durations):.2f}s, Avg: {sum(s_durations)/len(s_durations):.2f}s")

if __name__ == "__main__":
    main()
