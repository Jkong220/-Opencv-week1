"""
Demonstrates images as NumPy arrays, simple drawing and pixel access.
"""
import cv2
import numpy as np

def main():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)

    cv2.rectangle(img, (20, 20), (280, 120), (255, 0, 0), 2)
    cv2.circle(img, (150, 200), 50, (0, 0, 255), 2)

    px_before = img[100, 100].copy()
    img[100, 100] = (255, 255, 255)
    print("Pixel before:", px_before, "after:", img[100,100])

    cv2.imshow("Core Basics", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
