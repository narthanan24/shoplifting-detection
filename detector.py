"""
Video object detection module using YOLOv8.
Detects persons, backpacks, handbags, and bottles/items.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple


class Detector:
    """YOLOv8-based detector for persons and suspicious items."""
    
    # COCO class IDs we're interested in
    PERSON_CLASS_ID = 0
    BACKPACK_CLASS_ID = 24
    HANDBAG_CLASS_ID = 26
    BOTTLE_CLASS_ID = 39
    
    # Additional item-like classes (various objects that could be shoplifted)
    ITEM_CLASS_IDS = [24, 26, 39, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (default: yolov8n.pt)
        """
        self.model = YOLO(model_path)
        self.item_class_ids = set([self.BACKPACK_CLASS_ID, self.HANDBAG_CLASS_ID, 
                                   self.BOTTLE_CLASS_ID] + self.ITEM_CLASS_IDS)
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons and items in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections, each containing:
            - 'class_id': COCO class ID
            - 'class_name': Class name
            - 'bbox': [x1, y1, x2, y2] bounding box
            - 'confidence': Detection confidence
            - 'type': 'person', 'backpack', 'handbag', 'bottle', or 'item'
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                # Only keep relevant classes
                # Lower confidence threshold for items to catch more objects
                min_confidence = 0.2 if class_id != self.PERSON_CLASS_ID else 0.25
                
                if (class_id == self.PERSON_CLASS_ID or class_id in self.item_class_ids) and confidence >= min_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Determine type
                    if class_id == self.PERSON_CLASS_ID:
                        det_type = 'person'
                    elif class_id == self.BACKPACK_CLASS_ID:
                        det_type = 'backpack'
                    elif class_id == self.HANDBAG_CLASS_ID:
                        det_type = 'handbag'
                    elif class_id == self.BOTTLE_CLASS_ID:
                        det_type = 'bottle'
                    else:
                        det_type = 'item'
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'type': det_type
                    })
        
        return detections
    
    def get_persons(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only persons."""
        return [d for d in detections if d['type'] == 'person']
    
    def get_items(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only items (backpack, handbag, bottle, item)."""
        return [d for d in detections if d['type'] in ['backpack', 'handbag', 'bottle', 'item']]

