#!/usr/bin/env python3
"""
Fast evaluation runner.
Loads the pre-computed detections cache to run the tracking and heuristics simulation
on all 182 videos instantly (under 2 seconds) and generates the full evaluation report.
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add current dir to path
sys.path.append(str(Path(__file__).resolve().parent))
from tracker import ByteTracker
from scratch.run_fast_eval_100 import process_video

SHOPLIFTING_DIR = Path("archive/shoplifting")
NORMAL_DIR = Path("archive/normal")
OUTPUT = Path("evaluation_results/full_eval_summary.txt")

def main():
    cache_path = Path("evaluation_results/detections_cache.pkl")
    if not cache_path.exists():
        print(f"Error: Cache file {cache_path} not found.")
        print("Please run `python3 cache_detections.py` first to generate it.")
        return
        
    print("Loading precomputed detections cache...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    print(f"Evaluating {len(cache)} archive videos with optimized heuristics...")
    
    videos = []
    # Match the exact sorting order of full_eval.py
    for path in sorted(NORMAL_DIR.glob("normal-*.mp4")):
        videos.append((path.name, "normal", f"normal/{path.name}"))
    for path in sorted(SHOPLIFTING_DIR.glob("shoplifting-*.mp4")):
        videos.append((path.name, "shoplifting", f"shoplifting/{path.name}"))
        
    results = []
    for name, label, key in videos:
        if key in cache:
            events = process_video(key, cache[key])
            count = len(events)
        else:
            count = 0
            
        expected_detect = label == "shoplifting"
        detected = count > 0
        ok = detected == expected_detect
        results.append((name, label, count, ok, expected_detect))
        
    total = len(results)
    correct = sum(1 for r in results if r[3])
    normal_results = [r for r in results if r[1] == "normal"]
    shop_results = [r for r in results if r[1] == "shoplifting"]
    normal_ok = sum(1 for r in normal_results if r[3])
    shop_ok = sum(1 for r in shop_results if r[3])
    fp = [r for r in normal_results if r[2] > 0]
    fn = [r for r in shop_results if r[2] == 0]
    
    lines = [
        "FULL EVALUATION SUMMARY",
        "=" * 60,
        f"Total videos: {total}",
        f"Correct: {correct}/{total} ({100 * correct / total:.1f}%)",
        "",
        f"Normal videos:   {normal_ok}/{len(normal_results)} correct "
        f"(expected: 0 suspicious events)",
        f"  False positives ({len(fp)}): "
        + ", ".join(r[0] for r in fp) if fp else "none",
        "",
        f"Shoplifting videos: {shop_ok}/{len(shop_results)} correct "
        f"(expected: >=1 suspicious event)",
        f"  Missed ({len(fn)}): "
        + ", ".join(r[0] for r in fn) if fn else "none",
        "",
        "DETAILED RESULTS",
        "-" * 60,
        f"{'Status':<6} {'Video':<22} {'Label':<12} {'Events':>6} {'Expected':<10}",
        "-" * 60,
    ]
    for name, label, count, ok, expected in results:
        exp = "detect" if expected else "none"
        lines.append(
            f"{'OK' if ok else 'FAIL':<6} {name:<22} {label:<12} {count:>6} {exp:<10}"
        )
        
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    
    print("\n" + "="*50)
    print("FAST EVALUATION RESULTS")
    print("="*50)
    print(f"Total Correct: {correct}/{total} ({100 * correct / total:.2f}%)")
    print(f"Normal Videos Correct: {normal_ok}/{len(normal_results)}")
    print(f"Shoplifting Videos Correct: {shop_ok}/{len(shop_results)}")
    print(f"\nSaved detailed evaluation report to: {OUTPUT}")

if __name__ == "__main__":
    main()
