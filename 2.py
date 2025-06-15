import cv2
import numpy as np

def apply_average_filter(image_path, kernel_sizes):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found or invalid path.")

    # Dictionary to store filtered images
    filtered_images = {}

    for k in kernel_sizes:
        # Apply average filter using cv2.blur
        blurred = cv2.blur(img, (k, k))
        filtered_images[k] = blurred

    return img, filtered_images

def resize_for_display(image, max_width=800, max_height=800):
    h, w = image.shape
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

if __name__ == "__main__":
    image_path = "./1.jpg"
    kernel_sizes = [3, 10, 20]

    try:
        original, filtered_dict = apply_average_filter(image_path, kernel_sizes)

        # Resize and display original image
        original_resized = resize_for_display(original)
        cv2.imshow("Original Image", original_resized)

        # Resize and display each filtered image
        for k, filtered_img in filtered_dict.items():
            resized_img = resize_for_display(filtered_img)
            cv2.imshow(f"Average Filter {k}x{k}", resized_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
