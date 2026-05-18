import re

with open("main.py", "r") as f:
    content = f.read()

# 1. Add _check_overlap and remove _check_item_taken_from_shelf and _analyze_full_timeline_for_theft
overlap_func = """    def _check_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        \"\"\"Check if two bounding boxes overlap.\"\"\"
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)"""

content = re.sub(r'    def _check_item_taken_from_shelf.*?    def _check_person_exits_soon', overlap_func + '\n\n    def _check_person_exits_soon', content, flags=re.DOTALL)

# 2. Update process_video logic
items_split_code = """            # Split bags and objects
            bags = [d for d in items if d['type'] in ['backpack', 'handbag']]
            objects = [d for d in items if d['type'] not in ['backpack', 'handbag']]"""

content = content.replace(
    "            self.frame_items.append((frame_num, items))",
    items_split_code + "\n            self.frame_items.append((frame_num, items))"
)

interaction_code_old = """                    # Check for items near person while at shelf
                    nearby_items = self._find_nearby_items(person['bbox'], items, threshold=200.0)
                    if nearby_items:
                        self.person_item_interactions[track_id].append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'items_count': len(nearby_items),
                            'near_shelf': True
                        })"""

interaction_code_new = """                    # Check for items near person while at shelf
                    nearby_items = self._find_nearby_items(person['bbox'], objects, threshold=200.0)
                    
                    # Check bag interactions
                    bag_interaction = False
                    for bag in bags:
                        if self._check_overlap(person['bbox'], bag['bbox']):
                            for obj in objects:
                                if self._check_overlap(bag['bbox'], obj['bbox']):
                                    bag_interaction = True
                                    break
                    
                    if nearby_items or bag_interaction:
                        self.person_item_interactions[track_id].append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'items_count': len(nearby_items),
                            'near_shelf': True,
                            'bag_interaction': bag_interaction
                        })"""
content = content.replace(interaction_code_old, interaction_code_new)

# Replace Heuristics 2, 3, 4
heuristics_new = """                # Heuristic 4: Bag/Pocket Concealment & Dropped Items
                history = self.person_item_interactions[track_id]
                
                # Bag Concealment: Item touching bag for multiple frames
                bag_interactions = sum(1 for h in history[-30:] if h.get('bag_interaction', False))
                
                # Sustained Item Disappearance (Pocket):
                # Count dropped significantly and stayed dropped
                pocket_concealment = False
                if len(history) >= 30:
                    recent = history[-15:]
                    older = history[-45:-15] if len(history) >= 45 else history[-30:-15]
                    avg_older = sum(h.get('items_count', 0) for h in older) / len(older)
                    avg_recent = sum(h.get('items_count', 0) for h in recent) / len(recent)
                    if avg_older >= 1.0 and avg_recent <= avg_older - 0.7:
                        pocket_concealment = True
                
                if bag_interactions >= 5 or pocket_concealment:
                    start_time = max(0, timestamp - 3.0)
                    end_time = min(video_info['duration'], timestamp + 3.0)
                    existing = next((e for e in self.suspicious_events 
                                   if e['track_id'] == track_id and 
                                   abs(e['start_time'] - start_time) < 3.0), None)
                    if not existing:
                        reason = 'Item concealed in bag' if bag_interactions >= 5 else 'Item concealed in pocket'
                        self.suspicious_events.append({
                            'track_id': track_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'reason': reason
                        })"""

content = re.sub(r'                # Heuristic 2: Item taken from shelf.*?\(possible concealment/theft\)\'\n                            }\)', heuristics_new, content, flags=re.DOTALL)

# 4. Remove the post-process _analyze_full_timeline_for_theft call
content = re.sub(r'        # Post-process: Analyze full timeline for theft patterns.*?self\.suspicious_events\.append\(theft_event\)', '', content, flags=re.DOTALL)

# 5. Update default time-threshold
content = content.replace("parser.add_argument('--time-threshold', type=float, default=5.0,", "parser.add_argument('--time-threshold', type=float, default=15.0,")
content = content.replace("help='Time in seconds person must stay near shelves (default: 5.0)')", "help='Time in seconds person must stay near shelves (default: 15.0)')")

# Double threshold in Heuristic 1
content = content.replace("if person_near_shelf_time[track_id] >= self.time_near_shelf_threshold * 2:  # Double threshold", "if person_near_shelf_time[track_id] >= self.time_near_shelf_threshold:  # Use base threshold")

with open("main.py", "w") as f:
    f.write(content)
