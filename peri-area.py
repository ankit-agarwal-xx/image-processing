import cv2
import numpy as np

def read_image(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def find_perimeter_area(binary_image, width):
    # Assuming binary_image is a 1D array representing a binary image
    perimeter = 0
    area = 0

    # Iterate through each pixel in the binary image
    for i in range(len(binary_image)):
        if binary_image[i] == 1:
            area += 1
            # Check neighboring pixels to find perimeter
            if i % width != 0 and binary_image[i - 1] == 0:
                perimeter += 1
            if (i + 1) % width != 0 and binary_image[i + 1] == 0:
                perimeter += 1
            if i >= width and binary_image[i - width] == 0:
                perimeter += 1
            if i + width < len(binary_image) and binary_image[i + width] == 0:
                perimeter += 1

    return perimeter, area

# Example usage
file_path = 'path/to/binary_image.bin'
image = read_image(file_path)

# Assuming the width of the image is known or can be calculated
width = 100  # Replace this with the actual width of your image

perimeter, area = find_perimeter_area(image.flatten(), width)
print(f'Perimeter: {perimeter}, Area: {area}')