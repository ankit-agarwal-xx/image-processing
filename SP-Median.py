
import cv2
import random

# Read the image
img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise
noisy_img = img.copy()
num_salt = int(0.05 * img.size)  # 5% of pixels
num_pepper = int(0.05 * img.size)  # 5% of pixels

# Salt mode
for i in range(num_salt):
    x_coord = random.randint(0, img.shape[1] - 1)  # Width
    y_coord = random.randint(0, img.shape[0] - 1)  # Height
    noisy_img[y_coord, x_coord] = 255

# Pepper mode
for i in range(num_pepper):
    x_coord = random.randint(0, img.shape[1] - 1)  # Width
    y_coord = random.randint(0, img.shape[0] - 1)  # Height
    noisy_img[y_coord, x_coord] = 0

# Apply median filter
filtered_img = noisy_img.copy()
n = 3  # Kernel size
for i in range(n // 2, img.shape[0] - n // 2):
    for j in range(n // 2, img.shape[1] - n // 2):
        kernel = noisy_img[i-n//2:i+n//2+1, j-n//2:j+n//2+1]
        sorted_pixels = sorted(kernel.flatten())
        filtered_img[i, j] = sorted_pixels[len(sorted_pixels) // 2]

# Display the original, noisy, and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
