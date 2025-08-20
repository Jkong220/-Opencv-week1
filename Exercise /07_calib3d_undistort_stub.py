"""
Stub for undistortion API.
"""
import cv2
import numpy as np

def main():
    K = np.array([[800, 0, 320],[0, 800, 240],[0,0,1]], np.float32)
    dist = np.array([0.1, -0.05, 0, 0, 0], np.float32)

    img = np.full((480, 640, 3), 255, np.uint8)
    cv2.putText(img, "Distorted", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (640,480), 0.5)
    undist = cv2.undistort(img, K, dist, None, newK)

    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", undist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
