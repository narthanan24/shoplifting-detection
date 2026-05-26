#!/usr/bin/env python3
"""Evaluate all archive videos and write expected-results summary."""
import contextlib
import io
import sys
from pathlib import Path

from main import ShopliftingDetector

SHOPLIFTING_DIR = Path("archive/shoplifting")
NORMAL_DIR = Path("archive/normal")
OUTPUT = Path("evaluation_results/full_eval_summary.txt")


def eval_video(path: Path) -> int:
    detector = ShopliftingDetector(time_near_shelf_threshold=15.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        events, _ = detector.process_video(str(path))
    return len(events)


def main():
    videos = []
    for path in sorted(NORMAL_DIR.glob("normal-*.mp4")):
        videos.append((path, "normal"))
    for path in sorted(SHOPLIFTING_DIR.glob("shoplifting-*.mp4")):
        videos.append((path, "shoplifting"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(videos)

    for i, (path, label) in enumerate(videos, 1):
        print(f"[{i}/{total}] {path.name}...", flush=True)
        try:
            count = eval_video(path)
        except Exception as e:
            count = -1
            print(f"  ERROR: {e}", flush=True)

        expected_detect = label == "shoplifting"
        detected = count > 0
        ok = count >= 0 and detected == expected_detect
        results.append((path.name, label, count, ok, expected_detect))

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

    OUTPUT.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:20]))
    print(f"... full report: {OUTPUT}")
    print(f"\n{correct}/{total} correct")


if __name__ == "__main__":
    main()
