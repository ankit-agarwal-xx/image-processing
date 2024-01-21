import cv2
import numpy as np
import random
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def saltandpepper(img,density):
    threshold=1-density
    height,width=img.shape[:2]
    output=np.zeros(img.shape,np.uint8)
    for i in range (height):
        for j in range (width):
            possibility=random.random()
            if(possibility<density):
                output[i][j]=0
            elif(possibility>threshold):
                output[i][j]=255
            else:
                output[i][j]=img[i][j]
    return output

def median_filter(img,filter_size):
    height,width=img.shape
    output=np.zeros(img.shape,np.uint8)
    filter_array= [img[0][0]]*filter_size
    if filter_size==9:
        for j in range (height-1):
            for i in range (width-1):
                filter_array[0]=img[j-1,i-1]
                filter_array[1]=img[j,i-1]
                filter_array[2]=img[j+1,i-1]
                filter_array[3]=img[j-1,i]
                filter_array[4]=img[j,i]
                filter_array[5]=img[j+1,i]
                filter_array[6]=img[j-1,i+1]
                filter_array[7]=img[j,i+1]
                filter_array[8]=img[j+1,i+1]
                # Sorting
                filter_array.sort()
                output[j][i]=filter_array[4]
    return output

# Gausian def
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2
        + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

img=cv2.imread("images.jpeg")
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
saltandpepper_image=saltandpepper(gray_image,0.05)
median_filter_img=median_filter(saltandpepper_image,9)
cv2.imshow("window",median_filter_img)
cv2.waitKey(0)
# Gausian
filter_size=int(input("Give a odd int value as filter size:"))
sigma=float(input("Enter sigma value:"))
gaussian_filter_kernel=gaussian_kernel(filter_size,sigma)
filtered_image=convolve2d(
    saltandpepper_image,gaussian_filter_kernel,mode="same",boundary="symm"
)
plt.imshow(filtered_image,cmap="gray")
plt.show()
