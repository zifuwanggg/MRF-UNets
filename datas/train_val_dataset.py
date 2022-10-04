import cv2
import torch
import random

from utils import get_label_file, encode_label


class TrainValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train_image_list, val_image_list, transform, class_rgb_values):
        self.dataset = dataset
        self.train_image_list = train_image_list
        self.val_image_list = val_image_list
        self.transform = transform
        self.class_rgb_values = class_rgb_values
        
        
    def __getitem__(self, index):
        train_image_file = self.train_image_list[index]
        val_index = random.choice(range(len(self.val_image_list)))
        val_image_file = self.val_image_list[val_index]
        
        train_label_file = get_label_file(train_image_file, self.dataset)
        val_label_file = get_label_file(val_image_file, self.dataset)
            
        train_image = cv2.imread(train_image_file)
        val_image = cv2.imread(val_image_file)
        train_label_bgr = cv2.imread(train_label_file)
        val_label_bgr = cv2.imread(val_label_file)
        train_label = encode_label(train_label_bgr, self.class_rgb_values)
        val_label = encode_label(val_label_bgr, self.class_rgb_values)    

        train_transformed = self.transform(image=train_image, mask=train_label)
        val_transformed = self.transform(image=val_image, mask=val_label)
        train_image, train_label = train_transformed["image"], train_transformed["mask"]  
        val_image, val_label = val_transformed["image"], val_transformed["mask"]
        
        return train_image, train_label, val_image, val_label
        
        
    def __len__(self):
        return len(self.train_image_list)