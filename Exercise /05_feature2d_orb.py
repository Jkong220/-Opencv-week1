"""
ORB keypoints and descriptors.
"""
import cv2
import numpy as np

def main():
    img = np.full((300, 300), 255, dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, 0, 2)
    cv2.line(img, (90, 90), (210, 210), 0, 2)

    orb = cv2.ORB_create(nfeatures=300)
    kps, des = orb.detectAndCompute(img, None)
    print(f"Keypoints: {len(kps)}, Descriptors: {None if des is None else des.shape}")

    out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("ORB Features", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
  
