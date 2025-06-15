import cv2
import numpy as np

def reduce_intensity_levels(image_path, num_levels):
    if not (num_levels & (num_levels - 1) == 0 and num_levels <= 256):
        raise ValueError("Number of levels must be an integer power of 2 (e.g., 2, 4, 8, 16, ..., 256)")

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found or invalid path provided.")

    # Compute the interval size
    level_interval = 256 // num_levels

    # Quantize the image
    quantized_img = (img // level_interval) * level_interval

    return img, quantized_img

def resize_for_display(image, max_width=800, max_height=800):
    h, w = image.shape
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


if __name__ == "__main__":
    image_path = "./1.jpg"
    num_levels = 2  

    try:
        original, quantized = reduce_intensity_levels(image_path, num_levels)

        # Resize images for display
        original_resized = resize_for_display(original)
        quantized_resized = resize_for_display(quantized)

        # Show resized images
        cv2.imshow("Original Image", original_resized)
        cv2.imshow(f"Quantized Image ({num_levels} levels)", quantized_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)
