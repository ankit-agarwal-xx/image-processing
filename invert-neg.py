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
image_path = "images.jpeg"
output_path = "inverted_image.jpg"
invert_image(image_path, output_path)