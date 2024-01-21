import cv2
import numpy as np

def scale_image(image_path, scale_factor_x, scale_factor_y):
    # Read the image
    original_image = cv2.imread(image_path)

   
    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Get the dimensions of the original image
    height, width = original_image.shape[:2]

    
    scaling_matrix = np.array([[scale_factor_x, 0, 0], [0, scale_factor_y, 0]], dtype=np.float32)

  
    scaled_image = cv2.warpAffine(original_image, scaling_matrix, (int(width * scale_factor_x), int(height * scale_factor_y)))

   
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Scaled Image", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the image file
    image_path = r"images.jpeg"

    # Specify scaling factors for x and y axes
    scale_factor_x = 2.0  # Scaling factor for the x-axis
    scale_factor_y = 1.5  # Scaling factor for the y-axis

    # Perform scaling and display the images
    scale_image(image_path, scale_factor_x, scale_factor_y)
