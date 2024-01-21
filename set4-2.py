import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(file_path):
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array

def show_image(img_array):
    plt.imshow(img_array, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

def local_homogeneity(image, window_size):
    rows, cols = image.shape
    homogeneity_map = np.zeros((rows, cols))

    padded_image = np.pad(image, pad_width=window_size // 2, mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            window = padded_image[i:i+window_size, j:j+window_size]
            local_variance = np.var(window)
            homogeneity_map[i, j] = local_variance

    return homogeneity_map

if __name__ == "__main__":
    image_path = r"C:\Users\satis\images.jpeg"
    
    image = read_image(image_path)

    show_image(image)

    window_size = 3  
    homogeneity_map = local_homogeneity(image, window_size)

    plt.imshow(homogeneity_map, cmap='hot', interpolation='nearest')
    plt.title("Local Homogeneity Map")
    plt.colorbar()
    plt.axis('off')
    plt.show()
