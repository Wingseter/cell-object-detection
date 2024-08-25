from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # load model

    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # TODO Visualize Results