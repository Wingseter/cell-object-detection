from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt") # load a model
    path = model.export(format="onnx")  # export the model to ONNX format