# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image file as a numpy array
squirrel_image = cv2.imread('images.jpeg')

# Convert the image to grayscale if it has three channels
if squirrel_image.shape[-1] == 3:
    gray_image = np.dot(squirrel_image[...,:3], [0.28, 0.59, 0.11])
else:
    gray_image = squirrel_image

# This function applies the Roberts edge detection algorithm
def roberts(img):
    # Get the height and width of the image
    height, width = img.shape
    
    # Initialize the output image
    output = np.zeros_like(img)
    
    # Loop over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the pixel values in the 2x2 neighborhood
            p1 = img[i, j]
            p2 = img[i, j+1] if j < width - 1 else 0
            p3 = img[i+1, j] if i < height - 1 else 0
            p4 = img[i+1, j+1] if i < height - 1 and j < width - 1 else 0
            
            # Compute the gradients in x and y directions
            gx = p1 - p4
            gy = p2 - p3
            
            # Compute the gradient magnitude
            g = np.sqrt((gx**2) + (gy**2))
            
            # Assign the gradient value to the output image
            output[i, j] = g
    
    return output

# This function applies the Laplacian edge detection algorithm
def laplacian(img):
    # Get the height and width of the image
    height, width = img.shape
    
    # Initialize the output image
    output = np.zeros_like(img)
    
    # Loop over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the pixel values in the 3x3 neighborhood
            p1 = img[i-1, j-1] if i > 0 and j > 0 else 0
            p2 = img[i-1, j] if i > 0 else 0
            p3 = img[i-1, j+1] if i > 0 and j < width - 1 else 0
            p4 = img[i, j-1] if j > 0 else 0
            p5 = img[i, j]
            p6 = img[i, j+1] if j < width - 1 else 0
            p7 = img[i+1, j-1] if i < height - 1 and j > 0 else 0
            p8 = img[i+1, j] if i < height - 1 else 0
            p9 = img[i+1, j+1] if i < height - 1 and j < width - 1 else 0
            
            # Compute the Laplacian value
            l = p1 + p2 + p3 + p4 - 8 * p5 + p6 + p7 + p8 + p9
            
            # Clip the value to the range [0, 255]
            l = np.clip(l, 0, 255)
            
            # Assign the Laplacian value to the output image
            output[i, j] = l
    
    return output

# This function applies the Prewitt edge detection algorithm
def prewitt(img):
    # Get the height and width of the image
    height, width = img.shape
    
    # Initialize the output image
    output = np.zeros_like(img)
    
    # Loop over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the pixel values in the 3x3 neighborhood
            p1 = img[i-1, j-1] if i > 0 and j > 0 else 0
            p2 = img[i-1, j] if i > 0 else 0
            p3 = img[i-1, j+1] if i > 0 and j < width - 1 else 0
            p4 = img[i, j-1] if j > 0 else 0
            p6 = img[i, j+1] if j < width - 1 else 0
            p7 = img[i+1, j-1] if i < height - 1 and j > 0 else 0
            p8 = img[i+1, j] if i < height - 1 else 0
            p9 = img[i+1, j+1] if i < height - 1 and j < width - 1 else 0
            
            # Compute the gradients in x and y directions
            gx = p3 + p6 + p9 - p1 - p4 - p7
            gy = p1 + p2 + p3 - p7 - p8 - p9
            
            # Compute the gradient magnitude
            g = np.sqrt((gx**2) + (gy**2))
            
            # Assign the gradient value to the output image
            output[i, j] = g
    
    return output

# Apply the edge detection algorithms to the grayscale image
roberts_image = roberts(gray_image)
laplacian_image = laplacian(gray_image)
prewitt_image = prewitt(gray_image)

# Plot the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(141)
plt.title("Normal image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(142)
plt.title("Roberts image")
plt.imshow(roberts_image, cmap="gray")
plt.subplot(143)
plt.title("Laplacian image")
plt.imshow(laplacian_image, cmap="gray")
plt.subplot(144)
plt.title("Prewitt image")
plt.imshow(prewitt_image, cmap="gray")
plt.tight_layout()
plt.show()
