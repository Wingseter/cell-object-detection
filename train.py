from ultralytics import YOLO
from trainUtils import fix_all_seeds
import yaml
import os

if __name__ == "__main__":
    with open('Config/trainConfig.yaml', 'r') as f:
        config = yaml.safe_load(f)

    fix_all_seeds(config["seed"])

    model = YOLO(config["model"],)  # load a model 

    os.makedirs(config["SAVE_PATH"], exist_ok=True)

    model.train(data = config["yaml_path"],
                epochs = config["epochs"],
                imgsz = config["imgsz"],
                seed = config["seed"],
                batch = config["batch"],
                workers = config["workers"],
                project = config["SAVE_PATH"],
                patience = config["patience"]
                )
    
