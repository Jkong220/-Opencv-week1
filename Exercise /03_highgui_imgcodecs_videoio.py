"""
Load/save images and open a webcam stream.
"""
import cv2

def main():
    img = 255 * (cv2.UMat(200, 300, cv2.CV_8UC3).get())
    cv2.imwrite("sample_out.png", img)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
