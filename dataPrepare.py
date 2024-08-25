from dataUtils import *
from trainUtils import *
from sklearn.model_selection import train_test_split
import yaml 

if __name__ == "__main__":
    with open('Config/dataConfig.yaml', 'r') as f:
        config = yaml.safe_load(f)

    fix_all_seeds(config["seed"])
    print(f"Set Seed as {config['seed']}")


    # Load All folder
    fold_paths = list(Path(config["data_path"]).glob('*')) 
    merge_df = create_merge_df(fold_paths)
    print("Create merge Dataframe Finish...")

    # Add Width, height, Count
    count_df = calculate_image_width_height_count(merge_df)
    print("Calculate Width, height, count Finish... ")

    # Create directories and set paths
    train_image_path, train_label_path, val_image_path, val_label_path, test_image_path, test_label_path = mkdir_yolo_data(
        config["yolo_train_path"], 
        config["yolo_val_path"], 
        config["yolo_test_path"]
    )

    print(f"Create Folder: {train_image_path}")
    print(f"Create Folder: {train_label_path}")
    print(f"Create Folder: {val_image_path}")
    print(f"Create Folder: {val_label_path}")
    print(f"Create Folder: {test_image_path}")
    print(f"Create Folder: {test_label_path}")

    # Create YOLO YAML file
    create_yolo_yaml(config["yolo_train_path"], config["yolo_val_path"], config["yaml_path"])

    # Split DataFrame into train, validation, and test sets
    ratio = config["train_valid_test_ratio"]  # Example: [0.7, 0.2, 0.1]
    train_df, temp_df = train_test_split(count_df, test_size=(1 - ratio[0]), random_state=config['seed'])
    valid_df, test_df = train_test_split(temp_df, test_size=(ratio[2] / (ratio[1] + ratio[2])), random_state=config['seed'])

    # Prepare data for YOLO format
    train_cell_df = make_cell_dataframe(train_df)
    valid_cell_df = make_cell_dataframe(valid_df)
    test_cell_df = make_cell_dataframe(test_df)
    print("Create Cell Dataframe Finished...")

    # Create class dictionary
    class_dict = create_class_dict(count_df)

    # Map class labels to class numbers
    train_cell_df['class_num'] = train_cell_df['class'].apply(lambda x: class_dict[x])
    valid_cell_df['class_num'] = valid_cell_df['class'].apply(lambda x: class_dict[x])
    test_cell_df['class_num'] = test_cell_df['class'].apply(lambda x: class_dict[x])

    # Prepare YOLO datasets
    prepare_yolo_dataset(train_cell_df, train_image_path, train_label_path)  # Prepare train dataset
    prepare_yolo_dataset(valid_cell_df, val_image_path, val_label_path)      # Prepare validation dataset
    prepare_yolo_dataset(test_cell_df, test_image_path, test_label_path)     # Prepare test dataset

    print("All Process Finished")





    



