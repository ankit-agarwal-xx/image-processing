import matplotlib.pyplot as plt 
import numpy as np


def RGBtoGRAY(image):
    height, width, _ = image.shape 
    output = np.zeros((height, width)) 
    for i in range(height):
        for j in range(width):
            output[i, j] = image[i, j, 0]*0.28 + image[i, j, 1]*0.59 + image[i, j,2]*0.10
    return output


def NegativeImg(image):
    height, width = image.shape
    output = np.zeros((height, width)) 
    for i in range(height):
        for j in range(width):
            output[i, j] = 255-image[i, j] 
    return output

IMAGE_PATH = r"images2.jpeg" 
input_img = plt.imread(IMAGE_PATH)
grayscale_img = RGBtoGRAY(input_img) 
negative_grayscale_img = NegativeImg(grayscale_img) 
plt.figure(figsize=(12, 6))
plt.subplot(121) 
plt.title("GrayScale Image")
plt.imshow(grayscale_img, cmap="gray") 
plt.subplot(122)
plt.title("Negative GrayScale Image") 
plt.imshow(negative_grayscale_img, cmap="gray") 
plt.show()
