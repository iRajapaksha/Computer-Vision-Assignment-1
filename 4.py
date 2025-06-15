import cv2
import numpy as np

def block_average(image, block_size):
    h, w = image.shape[:2]
    
    # Ensure dimensions are multiples of block_size
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    img_cropped = image[:h_trim, :w_trim]

    output = img_cropped.copy()

    for i in range(0, h_trim, block_size):
        for j in range(0, w_trim, block_size):
            block = img_cropped[i:i+block_size, j:j+block_size]

            # Compute the mean of the block
            mean_value = np.mean(block, axis=(0, 1), dtype=np.uint8)

            # Fill the block with the mean value
            output[i:i+block_size, j:j+block_size] = mean_value

    return output

if __name__ == "__main__":
    image_path = "./1.jpg"  # Replace with your image path
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")

    block_sizes = [3, 5, 7]
    results = {}

    for b in block_sizes:
        results[b] = block_average(img, b)

    # Resize for display
    def resize(image, max_width=800, max_height=800):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            return cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    cv2.imshow("Original Image", resize(img))
    for b, processed_img in results.items():
        cv2.imshow(f"{b}x{b} Block Averaged", resize(processed_img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
