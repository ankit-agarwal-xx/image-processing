import cv2
import matplotlib.pyplot as plt

def find_centroid_area(binary_image_path):
    # Read the binary image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None, None

    # Assume the largest contour as the object of interest
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate centroid
    moments = cv2.moments(largest_contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # Calculate area
    area = cv2.contourArea(largest_contour)

    return (centroid_x, centroid_y), area

# Example usage
binary_image_path = 'images.jpeg'
centroid, area = find_centroid_area(binary_image_path)

if centroid is not None and area is not None:
    print(f"Centroid: ({centroid[0]}, {centroid[1]})")
    print(f"Area: {area}")

    # Display the image intensity values using matplotlib
    plt.imshow(cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('Binary Image')
    plt.show()
