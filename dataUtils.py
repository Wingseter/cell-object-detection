from pathlib import Path
from natsort import natsorted
from skimage.io import imread

import pandas as pd
import cv2
import shutil
import numpy as np

def check_list_maching(images, lables, binary): # check each image, label and binary 
    for im,la,bi in zip(images, lables, binary): 
        if im.with_suffix('').name == la.with_suffix('').name == bi.with_suffix('').name: 
            pass
        else: 
            assert print(f'Not matching same directory:{im}')

def create_merge_df(fold_paths):        
    merge_df = [] 

    for fold_path in fold_paths: # all split like blander brain cardia etc...
        meta_dict = {}
        img_type, organ = fold_path.name.split(' ',1) # (species ex; human, mouse), (organtype)
        if organ == 'fat (white and brown)_subscapula': organ = 'fat_subscapula' # name is too long
        img_pathes = natsorted(fold_path.glob('tissue images/*.png')) # sorting images
        labels_pathes = natsorted(fold_path.glob('label masks/*.tif')) # sorting labels
        binary_pathes = natsorted(fold_path.glob('mask binary/*.png')) # sorting binar
        
        meta_dict['Type'] = img_type # (human, mouse .. etc)
        meta_dict['organs'] = organ # organ type 
        meta_dict['img_path'] = img_pathes # image type 
        meta_dict['label_masks'] = labels_pathes # label type 
        meta_dict['binary_path'] = binary_pathes # binary type 
        merge_df.append(pd.DataFrame(meta_dict)) # merge it to data frame 

    
    df = pd.concat(merge_df).reset_index(drop=True)
    return df 

def create_class_dict(df: pd.DataFrame):
    class_dict = {cls_name : num for num, cls_name in enumerate(df['organs'].unique())} # number each organs 
    return class_dict

def calculate_image_width_height_count(df: pd.DataFrame):
    # counting cell and add dataframe
    df['count'], df['width'], df['height'] = 0, 0, 0
    for row in df.itertuples(): 
        mask = imread(row.label_masks)
        
        df.loc[row[0], 'width']  = mask.shape[-1]
        df.loc[row[0], 'height'] = mask.shape[-2]
        df.loc[row[0], 'count']  = mask.max()

    return df


def get_contour_bbox(msk): # bbox calculation
    """ Reference : https://www.kaggle.com/code/dschettler8845/train-sartorius-segmentation-eda-effdet-tf """
    """ Function to return the bounding box (tl, br) for a given mask """
    
    # Get contour(s) --> There should be only one
    assert msk.dtype == np.uint8 , "image type must uint8"
    
    cnts = cv2.findContours(msk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour, hierarchy = cnts
    
    if len(contour)==0:
        return None
    else:
        contour = contour[0]
    
    # Get extreme coordinates
    tl = (tuple(contour[contour[:, :, 0].argmin()][0])[0], 
          tuple(contour[contour[:, :, 1].argmin()][0])[1])
    br = (tuple(contour[contour[:, :, 0].argmax()][0])[0], 
          tuple(contour[contour[:, :, 1].argmax()][0])[1])
    
    return tl, br


## dataset reference : https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
def make_cell_dataframe(df):  # make bounding box 
    cell_list = []
    for row in df.itertuples(): 
        cell_dict = {'class'   :[], 'name'    :[], 'img_path':[],
                     'xmin'    :[], 'xmax'    :[], 'ymin'    :[], 'ymax'    :[],
                     'x_center':[], 'y_center':[], 'yolo_w'  :[], 'yolo_h'  :[]
                    }
        class_name = str(row.organs)
        mask = imread(row.label_masks)

        for i in np.unique(mask)[1:]:
            binary_img =np.where(mask==i, 1, 0).astype(np.uint8)
            min_coord, max_coord = get_contour_bbox(binary_img)

            xmin, ymin = min_coord
            xmax, ymax = max_coord
            
            cell_dict['img_path'].append(row.img_path)
            cell_dict['class'].append(class_name)
            cell_dict['name'].append(row.label_masks.with_suffix('').name)
            cell_dict['xmin'].append(xmin)
            cell_dict['xmax'].append(xmax)
            cell_dict['ymin'].append(ymin)
            cell_dict['ymax'].append(ymax)
            cell_dict['x_center'].append((xmin+xmax)/2/row.width)
            cell_dict['y_center'].append((ymin+ymax)/2/row.height)
            cell_dict['yolo_w'].append((xmax-xmin)/row.width)
            cell_dict['yolo_h'].append((ymax-ymin)/row.width)
        cell_list.append(pd.DataFrame(cell_dict))

    cell_df = pd.concat(cell_list)
    return cell_df 

def prepare_yolo_dataset(df, image_path, label_path): 
    unique_name = df['name'].unique()
    for image_name in unique_name: 
        name_df = df[df['name'] == image_name]

        label_txt = ''
        for coor_array in name_df[['class_num', 'x_center','y_center', 'yolo_w', 'yolo_h']].values: 
            coor_list = list(coor_array.reshape(-1).astype(str))
            coor_str = ' '.join(coor_list)
            # add string to label txt
            label_txt += f'{coor_str}\n'

        with open(label_path/(str(image_name)+'.txt'), 'w') as f:
            f.write(label_txt)

        shutil.copy2(name_df.iloc[0].img_path, image_path)

    
def create_yolo_yaml(train_path, val_path, save_path):
    # Edit yaml content
    yaml_content = f'''
    train: {train_path}
    val: {val_path}

    names:
        0:  muscle_tibia,
        1:  liver,
        2:  umbilical cord,
        3:  thymus,
        4:  lung,
        5:  epiglottis,
        6:  spleen,
        7:  fat_subscapula,
        8:  cardia,
        9:  salivory gland,
        10: melanoma,
        11: kidney,
        12: pylorus,
        13: jejunum,
        14: testis,
        15: tongue,
        16: cerebellum,
        17: oesophagus,
        18: heart,
        19: pancreas,
        20: brain,
        21: muscle,
        22: placenta,
        23: bladder,
        24: tonsile,
        25: rectum,
        26: femur,
        27: peritoneum
    '''

    with open(save_path, 'w') as f:
        f.write(yaml_content)


from pathlib import Path

def mkdir_yolo_data(train_path, val_path, test_path):
    """
    Make YOLO data's directories for train, validation, and test sets.
    
    Parameters
    ----------
    train_path: str
        Path for training data.
    val_path: str
        Path for validation data.
    test_path: str
        Path for testing data.
    
    Returns
    ----------
    train_image_path: str
        Path for images of training data.
    train_label_path: str
        Path for labels of training data.
    val_image_path: str
        Path for images of validation data.
    val_label_path: str
        Path for labels of validation data.
    test_image_path: str
        Path for images of test data.
    test_label_path: str
        Path for labels of test data.
    """
    train_image_path = Path(f'{train_path}/images')
    train_label_path = Path(f'{train_path}/labels')
    val_image_path = Path(f'{val_path}/images')
    val_label_path = Path(f'{val_path}/labels')
    test_image_path = Path(f'{test_path}/images')
    test_label_path = Path(f'{test_path}/labels')
    
    # Create Directory
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)
    test_image_path.mkdir(parents=True, exist_ok=True)
    test_label_path.mkdir(parents=True, exist_ok=True)
    
    return train_image_path, train_label_path, val_image_path, val_label_path, test_image_path, test_label_path

def export_df(df: pd.DataFrame, target_path):
    df.to_csv(target_path)