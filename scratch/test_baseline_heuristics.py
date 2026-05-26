import pickle
from collections import defaultdict

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)

video_data = cache['shoplifting/shoplifting-1.mp4']
frames = video_data['detections']
fps = video_data['metadata']['fps']
duration = video_data['metadata']['duration']

from tracker import ByteTracker
tracker = ByteTracker()
person_item_interactions = defaultdict(list)
person_near_shelf_time = defaultdict(float)
person_first_seen = {}
person_last_seen = {}
suspicious_events = []

def _check_overlap(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)

def _find_nearby_items(person_bbox, items, threshold=200.0):
    nearby = []
    for item in items:
        # Calculate distance
        x1_1, y1_1, x2_1, y2_1 = person_bbox
        x1_2, y1_2, x2_2, y2_2 = item['bbox']
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        if dist < threshold:
            nearby.append(item)
    return nearby

for frame_num, detections in enumerate(frames):
    timestamp = frame_num / fps
    tracked_persons = tracker.update(detections)
    
    items = [d for d in detections if d['type'] in ['backpack', 'handbag', 'bottle', 'item']]
    bags = [d for d in items if d['type'] in ['backpack', 'handbag']]
    objects = [d for d in items if d['type'] not in ['backpack', 'handbag']]
    
    for person in tracked_persons:
        track_id = person['track_id']
        if track_id not in person_first_seen:
            person_first_seen[track_id] = frame_num
            person_near_shelf_time[track_id] = 0.0
        person_last_seen[track_id] = frame_num
        
        person_near_shelf_time[track_id] += 1.0 / fps
        nearby_items = _find_nearby_items(person['bbox'], objects, threshold=200.0)
        
        bag_interaction = False
        item_in_pocket_zone = False
        
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            px1, py1, px2, py2 = person['bbox']
            margin_x = (px2 - px1) * 0.05
            margin_y = (py2 - py1) * 0.05
            
            is_inside = (ox1 >= px1 - margin_x and ox2 <= px2 + margin_x and 
                         oy1 >= py1 - margin_y and oy2 <= py2 + margin_y)
            person_center_x = (px1 + px2) / 2
            obj_center_x = (ox1 + ox2) / 2
            is_central = abs(obj_center_x - person_center_x) < (px2 - px1) * 0.3
            
            if is_inside and is_central:
                item_in_pocket_zone = True
                
            for bag in bags:
                if _check_overlap(person['bbox'], bag['bbox']):
                    if _check_overlap(bag['bbox'], obj['bbox']):
                        bag_interaction = True
        
        if nearby_items or bag_interaction or item_in_pocket_zone:
            person_item_interactions[track_id].append({
                'frame': frame_num,
                'timestamp': timestamp,
                'items_count': len(nearby_items),
                'bag_interaction': bag_interaction,
                'item_in_pocket_zone': item_in_pocket_zone
            })
            
    for person in tracked_persons:
        track_id = person['track_id']
        history = person_item_interactions[track_id]
        bag_interactions = sum(1 for h in history[-30:] if h.get('bag_interaction', False))
        
        pocket_concealment = False
        if len(history) >= 30:
            recent = history[-15:]
            older = history[-45:-15] if len(history) >= 45 else history[-30:-15]
            avg_older = sum(h.get('items_count', 0) for h in older) / len(older)
            avg_recent = sum(h.get('items_count', 0) for h in recent) / len(recent)
            
            if avg_older >= 1.0 and avg_recent <= avg_older - 0.7:
                if any(h.get('item_in_pocket_zone', False) for h in older[-15:]):
                    pocket_concealment = True
        
        if bag_interactions >= 5 or pocket_concealment:
            reason = 'Item concealed in bag' if bag_interactions >= 5 else 'Item concealed in pocket'
            print(f"Frame {frame_num} ({timestamp:.2f}s): Person {track_id} flagged for {reason} (bag_int={bag_interactions}, pocket={pocket_concealment})")
            
            existing = next((e for e in suspicious_events if e['track_id'] == track_id), None)
            if not existing:
                suspicious_events.append({'track_id': track_id, 'reason': reason})
