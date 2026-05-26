#!/usr/bin/env python3
"""Merge partial shard files into the main detections cache file."""
import pickle
from pathlib import Path

def main():
    dir_path = Path("evaluation_results")
    combined = {}
    
    # Load and merge all cache_part_*.pkl files
    shard_files = sorted(dir_path.glob("cache_part_*.pkl"))
    if not shard_files:
        print("No shard files found in evaluation_results/")
        return
        
    for shard in shard_files:
        try:
            with open(shard, "rb") as f:
                data = pickle.load(f)
            combined.update(data)
            print(f"Loaded {len(data)} videos from {shard.name}")
        except Exception as e:
            print(f"Error loading {shard.name}: {e}")
            
    if not combined:
        print("No videos merged.")
        return
        
    output_path = dir_path / "detections_cache.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(combined, f)
        
    normals = sum(1 for k in combined.keys() if k.startswith("normal/"))
    shops = sum(1 for k in combined.keys() if k.startswith("shoplifting/"))
    
    print(f"\nSuccessfully merged {len(combined)} videos into {output_path}")
    print(f"  Normal videos: {normals}")
    print(f"  Shoplifting videos: {shops}")

if __name__ == "__main__":
    main()
