#!/usr/bin/env python3
import random
from PIL import Image
import numpy as np
from generate_dataset import generate_plan, draw_plan, draw_mask

random.seed(42)
rooms, openings, wall_midlines, wall_info, wall_t, num_rooms = generate_plan(1200, 1200)

img = Image.new("RGB", (1200, 1200), (255,255,255))
draw_plan(img, rooms, openings, wall_t, wall_info)
img.save("test_color_plan.png")

mask = Image.new("RGB", (1200, 1200), (0,0,0))
draw_mask(mask, rooms, openings, wall_t, wall_info)
mask.save("test_color_mask.png")

arr = np.array(img)
# Check what colors the dimension-related pixels have
# Look for any pure white (255,255,255) in the image
white_mask = (arr[:,:,0] == 255) & (arr[:,:,1] == 255) & (arr[:,:,2] == 255)
bg_mask = (arr[:,:,0] == 245) & (arr[:,:,1] == 245) & (arr[:,:,2] == 245)
black_mask = (arr[:,:,0] == 0) & (arr[:,:,1] == 0) & (arr[:,:,2] == 0)

print(f"Image size: {arr.shape}")
print(f"White (255,255,255) pixels in image: {np.sum(white_mask)}")
print(f"Background (245,245,245) pixels: {np.sum(bg_mask)}")
print(f"Pure black (0,0,0) pixels: {np.sum(black_mask)}")

mask_arr = np.array(mask)
white_mask_m = (mask_arr[:,:,0] == 255) & (mask_arr[:,:,1] == 255) & (mask_arr[:,:,2] == 255)
print(f"White (255,255,255) pixels in mask: {np.sum(white_mask_m)}")

# Check: do any non-bg, non-wall-clear pixels have white color in image?
wall_colors = set()
for y in range(min(50, arr.shape[0])):
    for x in range(min(50, arr.shape[1])):
        wall_colors.add(tuple(arr[y,x]))
print(f"Sample colors in top-left 50x50: {sorted(wall_colors)[:10]}")
