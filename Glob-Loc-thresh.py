# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# This code performs global and local thresholding on the image

# Read the image file as a numpy array
squirrel_image = cv2.imread('images.jpeg')

# Convert the image to grayscale if it has three channels
if squirrel_image.shape[-1] == 3:
    gray_image = np.dot(squirrel_image[...,:3], [0.28, 0.59, 0.11])
else:
    gray_image = squirrel_image

# Global thresholding
global_thresholding = 127
global_thresholded_image = ((gray_image >= global_thresholding) * 255)

# Local thresholding
block_size = 10
constant_err = 3
height, width = gray_image.shape
local_thresholded_image = np.zeros_like(gray_image, dtype=np.uint8)

# Loop over each block in the image
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        # Get the block from the image
        block = gray_image[i:i+block_size, j:j+block_size]
        
        # Compute the mean of the block
        block_mean = np.mean(block)
        
        # Compute the threshold for the block
        threshold = block_mean - constant_err
        
        # Apply the threshold to the block
        local_thresholded_image[i:i+block_size, j:j+block_size] = ((block >= threshold) * 255)

# Plot the original and thresholded images
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title("Original image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(132)
plt.title("Global thresholding")
plt.imshow(global_thresholded_image, cmap="gray")
plt.subplot(133)
plt.title("Local thresholding")
plt.imshow(local_thresholded_image, cmap="gray")
plt.tight_layout()
plt.show()
