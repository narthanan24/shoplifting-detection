#!/usr/bin/env python3
"""Cache YOLO detections for all archive videos."""
import os
import pickle
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent))
from detector import Detector

SHOPLIFTING_DIR = Path("archive/shoplifting")
NORMAL_DIR = Path("archive/normal")
CACHE_PATH = Path("evaluation_results/detections_cache.pkl")


def main():
    # Make sure output directory exists
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Find all videos
    videos = []
    for path in sorted(NORMAL_DIR.glob("normal-*.mp4")):
        videos.append((path, "normal"))
    for path in sorted(SHOPLIFTING_DIR.glob("shoplifting-*.mp4")):
        videos.append((path, "shoplifting"))

    print(f"Found {len(videos)} videos to process.")

    # Initialize model
    print("Initializing YOLOv8n model on mps...")
    model = YOLO("yolov8n.pt")
    
    # Pre-configure coco classes we need
    # Person (0), Backpack (24), Handbag (26), Bottle (39) + other item classes
    item_class_ids = set([24, 26, 39] + Detector.ITEM_CLASS_IDS)
    allowed_classes = {0}.union(item_class_ids)

    # Check if cache exists, load it
    cache = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded existing cache with {len(cache)} videos.")
        except Exception as e:
            print(f"Error loading existing cache: {e}. Starting fresh.")

    total = len(videos)
    batch_size = 32

    for idx, (path, label) in enumerate(videos, 1):
        rel_name = f"{label}/{path.name}"
        if rel_name in cache:
            print(f"[{idx}/{total}] Skipping {rel_name} (already cached)")
            continue

        print(f"[{idx}/{total}] Processing {rel_name}...", flush=True)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  Error: Could not open {path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        frames_detections = []
        frame_buffer = []
        frame_indices = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            frame_indices.append(frame_idx)
            frame_idx += 1

            if len(frame_buffer) == batch_size:
                # Process batch
                results = model(frame_buffer, verbose=False, device="mps")
                for r_idx, result in zip(frame_indices, results):
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
                
                frame_buffer = []
                frame_indices = []

        # Process remaining frames
        if frame_buffer:
            results = model(frame_buffer, verbose=False, device="mps")
            for r_idx, result in zip(frame_indices, results):
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

        cap.release()

        # Save to cache
        cache[rel_name] = {
            'metadata': {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            },
            'detections': frames_detections
        }

        # Periodic save
        if idx % 10 == 0 or idx == total:
            print(f"Saving cache to {CACHE_PATH}...", flush=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.load = pickle.load # dummy test
                pickle.dump(cache, f)

    print("Caching completed successfully!")


if __name__ == "__main__":
    main()
