import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def global_threshold_binary(image, threshold=100):
    # Apply global threshold to generate binary image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def display_images(original_image, binary_image):
    # Display the original and binary images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image (Threshold=100)')

    plt.show()

# Example usage
file_path = "path/to/your/image.jpg"
original_image = read_image(file_path)
binary_image = global_threshold_binary(original_image)
display_images(original_image, binary_image)