import cv2
import numpy as np

def find_contours(binary_image):
    contours = []

    height, width = binary_image.shape

    
    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 255:
                contour = []
                contour.append((x, y))
                binary_image[y, x] = 0
                stack = []
                stack.append((x, y))
                while stack:
                    px, py = stack.pop()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] == 255:
                            contour.append((nx, ny))
                            binary_image[ny, nx] = 0
                            # Append the neighbor to the stack
                            stack.append((nx, ny))
                # Append the current contour to the list of contours
                contours.append(contour)

    # Return the list of contours
    return contours

def calculate_moments(contour):
    
    moments = {}

    # Calculate the zeroth order moment (area)
    moments["m00"] = len(contour)

    # Calculate the first order moments (sum of x and y coordinates)
    moments["m10"] = sum(x for x, y in contour)
    moments["m01"] = sum(y for x, y in contour)

    # Calculate the second order moments (sum of squared x and y coordinates)
    moments["m20"] = sum(x * x for x, y in contour)
    moments["m02"] = sum(y * y for x, y in contour)
    moments["m11"] = sum(x * y for x, y in contour)

    # Calculate the third order moments (sum of cubed x and y coordinates)
    moments["m30"] = sum(x * x * x for x, y in contour)
    moments["m03"] = sum(y * y * y for x, y in contour)
    moments["m21"] = sum(x * x * y for x, y in contour)
    moments["m12"] = sum(x * y * y for x, y in contour)

    # Return the dictionary of moments
    return moments

def find_centroid_perimeter(binary_image_path):
    # Read the binary image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the binary image
    contours = find_contours(binary_image)

    if not contours:
        print("No contours found.")
        return None, None

    # Assume the largest contour as the object of interest
    largest_contour = max(contours, key=len)

    # Calculate centroid
    moments = calculate_moments(largest_contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    
    perimeter = 0

    for i in range(len(largest_contour)):
        x1, y1 = largest_contour[i]
        x2, y2 = largest_contour[(i + 1) % len(largest_contour)]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        perimeter += distance

    return (centroid_x, centroid_y), perimeter

binary_image_path = r"images.jpeg"
centroid, perimeter = find_centroid_perimeter(binary_image_path)

if centroid is not None and perimeter is not None:
    print(f"Centroid: ({centroid[0]}, {centroid[1]})")
    print(f"Perimeter: {perimeter}")
