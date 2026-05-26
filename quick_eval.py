#!/usr/bin/env python3
"""Quick evaluation: count suspicious events per video without clip extraction."""
import sys
from pathlib import Path

from main import ShopliftingDetector

SHOPLIFTING_DIR = Path("archive/shoplifting")
NORMAL_DIR = Path("archive/normal")


def eval_video(path: Path) -> int:
    detector = ShopliftingDetector(time_near_shelf_threshold=15.0)
    events, _ = detector.process_video(str(path))
    return len(events)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        shoplifting = sorted(SHOPLIFTING_DIR.glob("shoplifting-*.mp4"))
        normal = sorted(NORMAL_DIR.glob("normal-*.mp4"))
        pairs = list(zip(shoplifting, normal))
    else:
        max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        pairs = []
        for i in range(1, max_n + 1):
            s = SHOPLIFTING_DIR / f"shoplifting-{i}.mp4"
            n = NORMAL_DIR / f"normal-{i}.mp4"
            if s.exists() and n.exists():
                pairs.append((s, n))

    results = []
    for shop_path, normal_path in pairs:
        for path, label in [(shop_path, "shoplifting"), (normal_path, "normal")]:
            if not path.exists():
                continue
            n = eval_video(path)
            expected = label == "shoplifting"
            detected = n > 0
            ok = detected == expected
            results.append((path.name, label, n, ok))
            status = "OK" if ok else "FAIL"
            print(f"{status} {path.name}: {n} event(s) (expected {'detect' if expected else 'none'})")

    fails = [r for r in results if not r[3]]
    print(f"\n{len(results) - len(fails)}/{len(results)} correct")
    if fails:
        print("Failures:")
        for name, label, n, _ in fails:
            print(f"  {name} ({label}): {n} events")


if __name__ == "__main__":
    main()
