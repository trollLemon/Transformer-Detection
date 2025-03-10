import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import os
import h5py
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from config import *
import albumentations as A
import numpy as np
import albumentations as A
from albumentations import BboxParams


# Define the augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.ColorJitter(p=0.4)
], bbox_params=BboxParams(format='coco'))


num_augmentations = 2



class CocoViTDataset(Dataset): 
    def __init__(self, root_dir, annotations_file, transform=None):
            self.coco = CocoDetection(root=root_dir, annFile=annotations_file)
            self.transform = transform

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, targets = self.coco[idx]

        if self.transform:
            img = self.transform(img)

        boxes = []
        labels = []

        for target in targets:
            bbox = target['bbox']
            boxes.append(bbox)

            category_id = target['category_id']
            labels.append(category_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, boxes, labels  

class HDF5ViTDataset(Dataset):
    
    def __init__(self, hdf5, transform = None):
        self.dataset = hdf5
        
        self.images = self.dataset['images']
        self.bboxs  = self.dataset['bboxes']
        self.labels = self.dataset['class_labels']
        self.transform = transform 
    def __len__(self):
        return len(self.images)
         
    def __getitem__(self, idx):
        
        image, bbox, label = self.images[idx], self.bboxs[idx], self.labels[idx]
        
        if self.transform:
            pil_image = Image.fromarray(image)
            image, bbox  = self.transform(pil_image, bbox)
            image = np.array(image)

        image, bbox, label = torch.tensor(image), torch.tensor(bbox), torch.tensor(label)

        return (image, bbox, label)




def init_dataset(group, dataset_size):
    
    group.create_dataset('bboxes', shape=(dataset_size, 4), dtype=np.float32)
    group.create_dataset('class_labels', shape=(dataset_size, 1), dtype=np.int64)
    group.create_dataset('images', shape=(dataset_size, 3, 224, 224), dtype=np.int64)
    
def init_hdf5(dataset_file_path, train_dataset_size, val_dataset_size):
    
    hdf5 = h5py.File(dataset_file_path, 'a')

    train = hdf5.create_group("train")
    init_dataset(train, train_dataset_size)
    val   = hdf5.create_group("val")
    init_dataset(val, val_dataset_size)
    
def create_hdf5_dataset(dataset_file_path):
 
       
    coco_train_set = CocoDetection(root=train_set, annFile=train_annotation_path)
    train_dataset_size = len(coco_train_set) * ( num_augmentations  + 1 )
    
    coco_val_set = CocoDetection(root=val_set, annFile=val_annotation_path)
    val_dataset_size = len(coco_val_set)

    if not os.path.isfile(dataset_file_path):
        init_hdf5(dataset_file_path, train_dataset_size, val_dataset_size)
 
    hdf5 = h5py.File(dataset_file_path, 'a')   
    print("Adding train images to dataset")
    add_to_dataset(hdf5, "train", coco_train_set, True)
    
    print("Adding validation images to dataset")
    add_to_dataset(hdf5, "val", coco_val_set, False)

def add_to_dataset(hdf5, group, coco, aug):
  

    #hdf5_path = os.path.join(output_path, dataset_file_name)
    
    #if not os.path.isfile(hdf5_path):
    #    init_hdf5(hdf5_path, dataset_size)
    no_target_count = 0

    # Initialize iteration counter
    itr = 0

    #hdf5 = h5py.File(hdf5_path, 'a')
    group = hdf5[group]

    invalid_count = 0

    valid_category_ids = list(range(1,81))
    for image, targets in tqdm(coco, desc="preprocessing data", unit="datapoint"):
        
        if len(targets) == 0:
            no_target_count += 1
            continue

        largest_box_index = 0
        largest_box_area = 0
        for i, target in enumerate(targets):
            bbox = target['bbox']
            area = bbox[2] * bbox[3]  # width * height
            if area > largest_box_area:
                largest_box_area = area
                largest_box_index = i
        
        # Extract the largest bounding box and label
        best_target = targets[largest_box_index]
        bbox_orig = best_target['bbox']
        category_id = best_target['category_id']
        
        if category_id not in valid_category_ids:
            invalid_count += 1

        for aug_idx in range(num_augmentations + 1):  # +1 for the original image
                        
            if aug_idx > 0:
                # Apply random augmentation
                transformed = transform(image=np.array(image), bboxes=[list(bbox_orig) + [category_id]])
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                bbox = transformed_bboxes[0][:4]  # Update bounding box
            else:
                transformed_image = np.array(image)
                bbox = best_target['bbox']
            
            # Process the transformed image and bounding box
            resized_image = Image.fromarray(transformed_image).resize((224, 224))
            bbox_x, bbox_y, bbox_w, bbox_h = bbox

            # Scale bounding box
            original_height, original_width = image.size
            x_scale = 224 / original_width
            y_scale = 224 / original_height

            bbox_x = int(bbox_x * x_scale) / 224
            bbox_y = int(bbox_y * y_scale) / 224
            bbox_w = int(bbox_w * x_scale) / 224
            bbox_h = int(bbox_h * y_scale) / 224

            bbox = (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h)

            box = np.array(bbox, dtype=np.float32)
            class_label = np.array(category_id, dtype=np.int64)
            image_np = np.array(resized_image)

            image_np = np.moveaxis(image_np, 2, 0)
            image_np = image_np / 255.0  # normalize

            group['bboxes'][itr] = box
            group['class_labels'][itr] = class_label
            group['images'][itr] = image_np
            itr += 1

            if not aug and aug_idx == 0:
                break

    print(f'{invalid_count} images were removed due to invalid classes (segmentation tasks)')
    print(f'{no_target_count} images were removed due to no target')

    
