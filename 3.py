import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Rotates the image around its center by the specified angle.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the size of the new bounding box
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take translation into account
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated

if __name__ == "__main__":
    image_path = "./1.jpg"  # Replace with your image path
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")

    # Rotate by 45 degrees (arbitrary)
    rotated_45 = rotate_image(img, 45)

    # Rotate by 90 degrees (built-in)
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Display the images (resized to fit)
    def resize(image, max_width=800, max_height=800):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            return cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    cv2.imshow("Original Image", resize(img))
    cv2.imshow("Rotated 45 Degrees", resize(rotated_45))
    cv2.imshow("Rotated 90 Degrees", resize(rotated_90))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
