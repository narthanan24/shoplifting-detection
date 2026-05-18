import re

with open("main.py", "r") as f:
    content = f.read()

# Replace the interaction check block
old_interaction = """                    # Check bag interactions
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

new_interaction = """                    # Check bag interactions and pocket zone
                    bag_interaction = False
                    item_in_pocket_zone = False
                    
                    for obj in objects:
                        # Check pocket zone (item completely inside person's bounding box)
                        ox1, oy1, ox2, oy2 = obj['bbox']
                        px1, py1, px2, py2 = person['bbox']
                        
                        # Add a small margin to account for YOLO bounding box tightness
                        margin_x = (px2 - px1) * 0.05
                        margin_y = (py2 - py1) * 0.05
                        
                        is_inside = (ox1 >= px1 - margin_x and ox2 <= px2 + margin_x and 
                                     oy1 >= py1 - margin_y and oy2 <= py2 + margin_y)
                                     
                        # Also require it to be centrally located to avoid edge holding
                        person_center_x = (px1 + px2) / 2
                        obj_center_x = (ox1 + ox2) / 2
                        is_central = abs(obj_center_x - person_center_x) < (px2 - px1) * 0.3
                        
                        if is_inside and is_central:
                            item_in_pocket_zone = True
                            
                        # Check bag
                        for bag in bags:
                            if self._check_overlap(person['bbox'], bag['bbox']):
                                if self._check_overlap(bag['bbox'], obj['bbox']):
                                    bag_interaction = True
                    
                    if nearby_items or bag_interaction or item_in_pocket_zone:
                        self.person_item_interactions[track_id].append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'items_count': len(nearby_items),
                            'near_shelf': True,
                            'bag_interaction': bag_interaction,
                            'item_in_pocket_zone': item_in_pocket_zone
                        })"""

content = content.replace(old_interaction, new_interaction)

# Replace the heuristic 4 block
old_heuristic = """                # Sustained Item Disappearance (Pocket):
                # Count dropped significantly and stayed dropped
                pocket_concealment = False
                if len(history) >= 30:
                    recent = history[-15:]
                    older = history[-45:-15] if len(history) >= 45 else history[-30:-15]
                    avg_older = sum(h.get('items_count', 0) for h in older) / len(older)
                    avg_recent = sum(h.get('items_count', 0) for h in recent) / len(recent)
                    if avg_older >= 1.0 and avg_recent <= avg_older - 0.7:
                        pocket_concealment = True"""

new_heuristic = """                # Sustained Item Disappearance (Pocket):
                # Count dropped significantly AND it was in pocket zone before dropping
                pocket_concealment = False
                if len(history) >= 30:
                    recent = history[-15:]
                    older = history[-45:-15] if len(history) >= 45 else history[-30:-15]
                    avg_older = sum(h.get('items_count', 0) for h in older) / len(older)
                    avg_recent = sum(h.get('items_count', 0) for h in recent) / len(recent)
                    
                    if avg_older >= 1.0 and avg_recent <= avg_older - 0.7:
                        # Strongly require that it was in the pocket zone recently
                        if any(h.get('item_in_pocket_zone', False) for h in older[-15:]):
                            pocket_concealment = True"""

content = content.replace(old_heuristic, new_heuristic)

with open("main.py", "w") as f:
    f.write(content)
