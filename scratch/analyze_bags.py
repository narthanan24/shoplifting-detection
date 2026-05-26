#!/usr/bin/env python3
import pickle

with open("evaluation_results/detections_cache.pkl", "rb") as f:
    cache = pickle.load(f)

normal_with_bag = 0
shop_with_bag = 0
normal_total = 0
shop_total = 0

for key, video_data in cache.items():
    is_shop = key.startswith("shoplifting/")
    if is_shop:
        shop_total += 1
    else:
        normal_total += 1
        
    has_bag = False
    for frame in video_data['detections']:
        bags = [d for d in frame if d['type'] in ['backpack', 'handbag']]
        if bags:
            has_bag = True
            break
            
    if has_bag:
        if is_shop:
            shop_with_bag += 1
        else:
            normal_with_bag += 1

print(f"Normal videos with bag: {normal_with_bag}/{normal_total} ({100*normal_with_bag/normal_total:.1f}%)")
print(f"Shoplifting videos with bag: {shop_with_bag}/{shop_total} ({100*shop_with_bag/shop_total:.1f}%)")
