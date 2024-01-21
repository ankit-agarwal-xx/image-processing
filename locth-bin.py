import cv2

def read_image(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def local_threshold(image, block_size=10):
    # Ensure block_size is an odd number greater than 1
    block_size = max(3, block_size)  # Ensure block_size is at least 3
    block_size = block_size + 1 if block_size % 2 == 0 else block_size

    # Use OpenCV's adaptiveThreshold function for local thresholding
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2
    )
    return binary_image

# Example usage
file_path = 'path/to/your/image.png'
image = read_image(file_path)

# Check if the image is successfully loaded
if image is not None:
    # Set block_size to 10
    block_size = 10

    binary_image = local_threshold(image, block_size)

    # Display the original and binary images using OpenCV
    cv2.imshow('Original Image', image)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error loading the image.")

