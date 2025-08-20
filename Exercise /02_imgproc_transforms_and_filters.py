"""
Resize, rotate, flip, blur, and edge detect.
"""
import cv2
import numpy as np

def rotate_image(image, angle_degrees):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_degrees, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def main():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (220, 140), (0, 0, 0), -1)

    resized = cv2.resize(img, (150, 150))
    rotated = rotate_image(img, 30)
    flipped = cv2.flip(img, 1)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    edges = cv2.Canny(img, 50, 150)

    cv2.imshow("Original", img)
    cv2.imshow("Resized", resized)
    cv2.imshow("Rotated", rotated)
    cv2.imshow("Flipped", flipped)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
  
