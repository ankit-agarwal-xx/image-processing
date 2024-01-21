import matplotlib.pyplot as plt 
import numpy as np

def RGBtoGRAY(image):
    height, width, _ = image.shape 
    output = np.zeros((height, width)) 
    for i in range(height):
        for j in range(width):
            output[i, j] = image[i, j, 0]*0.28 +image[i, j, 1]*0.59 + image[i, j, 2]*0.10
    return output

IMAGE_PATH = r"images.jpeg"
input_img = plt.imread(IMAGE_PATH) 
grayscale_img = RGBtoGRAY(input_img) 
plt.figure(figsize=(12, 6)) 
plt.subplot(121) 
plt.title("Original Image") 
plt.imshow(input_img) 
plt.subplot(122) 
plt.title("GrayScale Image")
plt.imshow(grayscale_img, cmap="gray") 
plt.show()
