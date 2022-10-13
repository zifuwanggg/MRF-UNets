import cv2
import torch

from datas.data_utils import get_label_file, encode_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_list, transform, class_rgb_values):
        self.dataset = dataset
        self.image_list = image_list
        self.transform = transform
        self.class_rgb_values = class_rgb_values
    
    
    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = get_label_file(image_file, self.dataset)
        
        if self.dataset in ['land', 'road', 'building']:
            image = cv2.imread(image_file) 
        else:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            
        label_bgr = cv2.imread(label_file)
        label = encode_label(label_bgr, self.class_rgb_values)

        transformed = self.transform(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"]   
        
        return image, label
        
        
    def __len__(self):
        return len(self.image_list)