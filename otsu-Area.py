import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('images.jpeg')
if img.shape[-1] == 3:
    img = np.dot(img[...,:3], [0.28, 0.59, 0.11])

# Local block_size=10 constant_err=3
height, width = img.shape
block_size = 10 
constant_err = 3
local_thresholding = np.zeros_like(img, dtype=np.uint8)
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        block = img[i:i+block_size, j:j+block_size]
        block_mean = np.mean(block)
        threshold = block_mean - constant_err
        local_thresholding[i:i+block_size, j:j+block_size] = (block >= threshold) * 255

# total pixel
height, width = local_thresholding.shape
total_pixel = height * width

# White pixel
white_count = 0
for i in range(height):
    for j in range(width):
        if (local_thresholding[i, j] == 255):
            white_count = white_count + 1
white_percentage = white_count * 100 / total_pixel
print("Total percentage is:", white_percentage)
