#!/usr/bin/env python3
"""Cache YOLO detections for all archive videos in parallel on CPU with checkpointing and thread optimization."""
import os
import pickle
import sys
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from multiprocessing import Process

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent))
from detector import Detector

SHOPLIFTING_DIR = Path("archive/shoplifting")
NORMAL_DIR = Path("archive/normal")
CACHE_PATH = Path("evaluation_results/detections_cache.pkl")


def worker_task(worker_id: int, videos_subset: list, output_shard_path: Path):
    # Optimize PyTorch threading to avoid contention on multi-core CPU
    torch.set_num_threads(1)
    
    print(f"[Worker {worker_id}] Starting, processing {len(videos_subset)} videos...")
    
    # Load existing shard cache if it exists
    shard_cache = {}
    if output_shard_path.exists():
        try:
            with open(output_shard_path, "rb") as f:
                shard_cache = pickle.load(f)
            print(f"[Worker {worker_id}] Loaded {len(shard_cache)} completed videos from shard. Resuming...")
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading shard {output_shard_path}: {e}. Starting fresh.")

    model = YOLO("yolov8n.pt")
    
    # Pre-configure coco classes we need
    item_class_ids = set([24, 26, 39] + Detector.ITEM_CLASS_IDS)
    allowed_classes = {0}.union(item_class_ids)

    total = len(videos_subset)

    for idx, (path, label) in enumerate(videos_subset, 1):
        rel_name = f"{label}/{path.name}"
        if rel_name in shard_cache:
            # Skip already completed
            continue

        t_start = time.time()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"[Worker {worker_id}] Error: Could not open {rel_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        frames_detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process single frame
            results = model(frame, verbose=False, device="cpu")
            result = results[0]
            
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in allowed_classes:
                        confidence = float(box.conf[0])
                        min_confidence = 0.2 if class_id != 0 else 0.25
                        if confidence >= min_confidence:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Type determination
                            if class_id == 0:
                                det_type = 'person'
                            elif class_id == 24:
                                det_type = 'backpack'
                            elif class_id == 26:
                                det_type = 'handbag'
                            elif class_id == 39:
                                det_type = 'bottle'
                            else:
                                det_type = 'item'

                            detections.append({
                                'class_id': class_id,
                                'class_name': result.names[class_id],
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'type': det_type
                            })
            frames_detections.append(detections)
            frame_idx += 1

        cap.release()

        shard_cache[rel_name] = {
            'metadata': {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            },
            'detections': frames_detections
        }
        
        t_duration = time.time() - t_start
        fps_actual = frame_count / t_duration if t_duration > 0 else 0
        print(f"[Worker {worker_id}] [{idx}/{total}] Completed {rel_name} in {t_duration:.1f}s ({fps_actual:.1f} fps)", flush=True)

        # Write shard cache after every single video
        with open(output_shard_path, "wb") as f:
            pickle.dump(shard_cache, f)

    print(f"[Worker {worker_id}] Finished and saved {len(shard_cache)} videos to {output_shard_path}")


def main():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Find all videos
    videos = []
    for path in sorted(NORMAL_DIR.glob("normal-*.mp4")):
        videos.append((path, "normal"))
    for path in sorted(SHOPLIFTING_DIR.glob("shoplifting-*.mp4")):
        videos.append((path, "shoplifting"))

    print(f"Found {len(videos)} videos to process.")

    num_workers = 4
    # Split videos into shards
    shards = [[] for _ in range(num_workers)]
    for i, item in enumerate(videos):
        shards[i % num_workers].append(item)

    processes = []
    shard_paths = []
    
    t0 = time.time()

    for i in range(num_workers):
        shard_path = CACHE_PATH.parent / f"cache_part_{i}.pkl"
        shard_paths.append(shard_path)
        p = Process(target=worker_task, args=(i, shards[i], shard_path))
        processes.append(p)
        p.start()

    # Wait for all processes
    for p in processes:
        p.join()

    # Combine shards
    combined_cache = {}
    for shard_path in shard_paths:
        if shard_path.exists():
            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
            combined_cache.update(shard_data)
            # Delete temp shard
            shard_path.unlink()

    # Save combined cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(combined_cache, f)

    total_time = time.time() - t0
    print(f"All workers finished! Combined cache has {len(combined_cache)} videos.")
    print(f"Saved to {CACHE_PATH}")
    print(f"Total time taken: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
