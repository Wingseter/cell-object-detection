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

    # Create Folder To Save Data
    train_image_path, train_label_path, val_image_path, val_label_path = mkdir_yolo_data(config["yolo_train_path"], config["yolo_val_path"])
    print(f"Create Folder: {train_image_path}")
    print(f"Create Folder: {train_label_path}")
    print(f"Create Folder: {val_image_path}")
    print(f"Create Folder: {val_label_path}")
    create_yolo_yaml(config["yolo_train_path"], config["yolo_val_path"], config["yaml_path"])

    # split DataFrame
    train_df, valid_df = train_test_split(count_df, test_size=config["train_valid_ratio"], random_state=config['seed'])

    # Make Data compatible for YOLO
    train_cell_df = make_cell_dataframe(train_df)
    valid_cell_df = make_cell_dataframe(valid_df)
    print("Create Cell Dataframe Finished...")

    class_dict = create_class_dict(count_df)

    train_cell_df['class_num'] = train_cell_df['class'].apply(lambda x : class_dict[x])
    valid_cell_df['class_num'] = valid_cell_df['class'].apply(lambda x : class_dict[x])

    prepare_yolo_dataset(train_cell_df, train_image_path, train_label_path) # train dataset prepare
    prepare_yolo_dataset(valid_cell_df, val_image_path, val_label_path) # valid dataset prepare 

    print("All Process Finish")

    # if Export CSV Need
    # export_df()






    



