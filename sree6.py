# import cv2
# import matplotlib.pyplot as plt

# def find_centroid_area(binary_image_path):
#     # Read the binary image
#     binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

#     # Find contours in the binary image
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         print("No contours found.")
#         return None, None

#     # Assume the largest contour as the object of interest
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Calculate centroid
#     moments = cv2.moments(largest_contour)
#     centroid_x = int(moments["m10"] / moments["m00"])
#     centroid_y = int(moments["m01"] / moments["m00"])

#     # Calculate area
#     area = cv2.contourArea(largest_contour)

#     return (centroid_x, centroid_y), area

# # Example usage
# binary_image_path = 'Cheesin.jpeg'
# centroid, area = find_centroid_area(binary_image_path)

# if centroid is not None and area is not None:
#     print(f"Centroid: ({centroid[0]}, {centroid[1]})")
#     print(f"Area: {area}")

#     # Display the image intensity values using matplotlib
#     plt.imshow(cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
#     plt.title('Binary Image')
#     plt.show()
# Python program to add salt-pepper noise and apply median filter

import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            rand = np.random.random()
            if rand < salt_prob:
                noisy_image[i, j] = 255  # Add salt noise
            elif rand > (1 - pepper_prob):
                noisy_image[i, j] = 0  # Add pepper noise

    return noisy_image

def apply_median_filter(image, kernel_size):
    filtered_image = np.copy(image)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            neighbors = []
            for m in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for n in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= m < height and 0 <= n < width:
                        neighbors.append(image[m, n])
            neighbors.sort()
            filtered_image[i, j] = neighbors[len(neighbors) // 2]

    return filtered_image

# Example usage:
original_image = cv2.imread('Cheesin.jpeg', cv2.IMREAD_GRAYSCALE)

# Convert the original image to a NumPy array
original_image_array = np.array(original_image)

# Add salt-pepper noise
salt_prob = 0.05
pepper_prob = 0.05
noisy_image = add_salt_pepper_noise(original_image_array, salt_prob, pepper_prob)

# Apply median filter with a 5x5 kernel
kernel_size = 5
filtered_image = apply_median_filter(noisy_image, kernel_size)

# Display the images using matplotlib
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(original_image_array, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(noisy_image, cmap='gray')
axs[1].set_title('Noisy Image')

axs[2].imshow(filtered_image, cmap='gray')
axs[2].set_title('Filtered Image')

plt.show()
