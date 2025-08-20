"""
Stub for DNN loading.
"""
import cv2
import numpy as np

def main():
    model_path = "model.onnx"
    try:
        net = cv2.dnn.readNet(model_path)
    except Exception:
        print("Provide a valid ONNX model.")
        return

    blob = cv2.dnn.blobFromImage(np.zeros((224,224,3), np.uint8), 1/255.0, (224,224))
    net.setInput(blob)
    out = net.forward()
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()
  
