import cv2
import numpy as np

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    # This function is the same as yours
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    salt_pixels = int(total_pixels * salt_prob)
    salt_coordinates = [np.random.randint(0, i - 1, salt_pixels) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1], :] = 255

    # Add pepper noise
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coordinates = [np.random.randint(0, i - 1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1], :] = 0

    return noisy_image

def apply_median_filter(image, kernel_size):
    # This function is modified to implement the median filter manually
    filtered_image = np.copy(image)
    # Get the height and width of the image
    height, width = image.shape[:2]
    # Get the radius of the kernel
    radius = kernel_size // 2
    # Loop over the image pixels, excluding the border pixels
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            # Get the neighboring pixels within the kernel size
            neighbors = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
            # Flatten the 2D array into 1D and sort it
            sorted_neighbors = np.sort(neighbors.ravel())
            # Find the median value and assign it to the filtered image
            median = sorted_neighbors[len(sorted_neighbors) // 2]
            filtered_image[y, x] = median
    return filtered_image

if __name__ == "__main__":
    # Load an image
    image_path = r"C:\Users\satis\images.jpeg"
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Could not read the image.")
    else:
        # Add salt and pepper noise
        salt_prob = 0.02  # Adjust as needed
        pepper_prob = 0.02  # Adjust as needed
        noisy_image = add_salt_pepper_noise(original_image, salt_prob, pepper_prob)

        # Apply median filter with a 5x5 kernel
        kernel_size = 5
        filtered_image = apply_median_filter(noisy_image, kernel_size)

        # Display the images
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Noisy Image", noisy_image)
        cv2.imshow("Filtered Image", filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
