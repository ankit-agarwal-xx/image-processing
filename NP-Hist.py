import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_binary_image(file_path):
    # Read the binary image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def apply_neighboring_padding(image):
    # Add neighboring pixel value padding to the image
    padded_image = np.pad(image, 1, mode='edge')
    return padded_image

def draw_histogram(image):
    # Draw histogram of the image
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    plt.plot(histogram, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Image')
    plt.show()

# Example usage
file_path = "path/to/your/binary_image.png"
binary_image = read_binary_image(file_path)
padded_image = apply_neighboring_padding(binary_image)
draw_histogram(padded_image)