# question 1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def invert_image(image_path, output_path):
    # Read the image
    img = Image.open(image_path)

    # Convert image to numpy array
    img_array = np.array(img)

    # Invert the image
    inverted_img_array = 255 - img_array

    # Convert back to Image
    inverted_img = Image.fromarray(inverted_img_array)

    # Save the inverted image
    inverted_img.save(output_path)

    # Display the original and inverted images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(inverted_img)
    axes[1].set_title('Inverted Image')
    axes[1].axis('off')

    plt.show()

# Example usage:
image_path = "Cheesin.jpeg"
output_path = "inverted_image.jpg"
invert_image(image_path, output_path)

# Question2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_entropy(image_path):
    # Read the image
    img = Image.open(image_path)

    # Convert image to grayscale
    img_gray = img.convert('L')

    # Convert image to numpy array
    img_array = np.array(img_gray)

    # Calculate histogram
    hist, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # Normalize histogram
    hist = hist / float(img_array.size)

    # Calculate entropy
    entropy = -np.sum([p * math.log2(p + 1e-10) for p in hist])

    return entropy, img_array

def display_image(img_array):
    # Display the image
    plt.imshow(img_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

# Example usage:
image_path = "Cheesin.jpeg"
entropy_value, image_array = calculate_entropy(image_path)
print(f"Entropy of the image: {entropy_value}")

# Display the image
display_image(image_array)
