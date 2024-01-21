
import cv2
import matplotlib.pyplot as plt

# Read the binary image
img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is successfully loaded
if img is None:
    print("Error: Unable to read the image. Make sure the file path is correct.")
    exit()

# Apply zero padding
padded_img = [[0 for _ in range(len(img[0]) + 2)] for _ in range(len(img) + 2)]
for i in range(len(img)):
    for j in range(len(img[0])):
        padded_img[i+1][j+1] = img[i][j]

# Draw histogram
histogram = [0]*256
for row in padded_img:
    for pixel in row:
        histogram[pixel] += 1

# Plot histogram
plt.bar(range(256), histogram)
plt.show()

# Display the original and padded images
cv2.imshow('Original Image', img)
cv2.imshow('Padded Image', np.array(padded_img, dtype=np.uint8))  # Convert to numpy array for display
cv2.waitKey(0)
cv2.destroyAllWindows()
