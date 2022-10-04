import os
import pickle
from glob import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datas.dataset import Dataset
from datas.train_val_dataset import TrainValDataset


def get_image_list(args):
    data_dir = os.path.join(args.data_dir, f"{args.dataset}/resized")
    
    if args.dataset in ['land', 'road', 'building']:
        train_image_list_path = f"pkls/{args.dataset}_train_image_list.pkl"
        test_image_list_path = f"pkls/{args.dataset}_test_image_list.pkl"  
        
        with open(train_image_list_path, 'rb') as f:
            train_image_list = pickle.load(f)
        with open(test_image_list_path, 'rb') as f:
            test_image_list = pickle.load(f)
        
        train_image_list = [os.path.join(data_dir, f) for f in train_image_list]
        test_image_list = [os.path.join(data_dir, f) for f in test_image_list]
        
        if args.supernet:
            assert args.dataset == 'land'
            length = len(test_image_list)
            val_image_list = train_image_list[-length:]
            train_image_list = train_image_list[:-length]
        else:
            val_image_list = []
        
        if args.dataset == 'land':
            if args.supernet:
                assert len(train_image_list) == 483
                assert len(val_image_list) == 160
                assert len(test_image_list) == 160 
            else:
                assert len(train_image_list) == 643
                assert len(val_image_list) == 0
                assert len(test_image_list) == 160 
            class_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
        elif args.dataset == 'road':
            assert len(train_image_list) == 4981
            assert len(val_image_list) == 0
            assert len(test_image_list) == 1245
            class_rgb_values = [[0, 0, 0], [255, 255, 255]]
        elif args.dataset == 'building':
            assert len(train_image_list) == 8475
            assert len(val_image_list) == 0
            assert len(test_image_list) == 2118
            class_rgb_values = [[0, 0, 0], [255, 255, 255]]
    
    elif args.dataset == 'chaos':
        train_case_id = [1, 2, 3, 5, 8, 10, 15, 19, 20, 21, 22, 31, 33, 34, 36, 38]
        test_case_id = [13, 32, 37, 39]
        assert len(train_case_id) == 16
        assert len(test_case_id) == 4
        
        train_image_list = [os.path.join(data_dir, str(case_id)) for case_id in train_case_id]
        test_image_list = [os.path.join(data_dir, str(case_id)) for case_id in test_case_id]
        train_image_list = expand_image_list(train_image_list, 'chaos')
        test_image_list = expand_image_list(test_image_list, 'chaos')
        val_image_list = []
        assert len(train_image_list) == 1552
        assert len(val_image_list) == 0
        assert len(test_image_list) == 365
        class_rgb_values = [[0, 0, 0], [63, 63, 63], [126, 126, 126], [189, 189, 189], [252, 252, 252]]
        
    elif args.dataset == 'promise':
        train_case_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 48]
        test_case_id = [13, 14, 15, 17, 25, 31, 37, 38, 47, 49]
        assert len(train_case_id) == 40
        assert len(test_case_id) == 10
        
        train_image_list = [os.path.join(data_dir, str(case_id)) for case_id in train_case_id]
        test_image_list = [os.path.join(data_dir, str(case_id)) for case_id in test_case_id]
        train_image_list = expand_image_list(train_image_list, 'promise')
        test_image_list = expand_image_list(test_image_list, 'promise')
        val_image_list = []
        assert len(train_image_list) == 1161
        assert len(val_image_list) == 0
        assert len(test_image_list) == 216
        class_rgb_values = [[0, 0, 0], [255, 255, 255]]
        
    return train_image_list, val_image_list, test_image_list, class_rgb_values


def expand_image_list(image_list, dataset):
    expanded_image_list = []
    for case in image_list:
        if dataset == "chaos":
            t1_in = glob(f"{case}/T1DUAL/DICOM_anon/InPhase/*")
            t1_out = glob(f"{case}/T1DUAL/DICOM_anon/OutPhase/*")
            t2 = glob(f"{case}/T2SPIR/DICOM_anon/*")
            expanded_image_list += t1_in + t1_out + t2
        elif dataset == "promise":
            expanded_image_list += glob(f"{case}/*_image.png")
    
    return expanded_image_list


def get_data_transform(dataset):
    size = 256
    
    if dataset == 'land':
        mean = [0.2822, 0.3795, 0.4089]
        std = [0.0882, 0.0996, 0.1350]
    elif dataset == 'road':
        mean = [0.2881, 0.3827, 0.4090]
        std = [0.1092, 0.1146, 0.1474] 
    elif dataset == 'building':
        mean = [0.2477, 0.2451, 0.2502]
        std = [0.1574, 0.1575, 0.1563]
    elif dataset == 'chaos':
        mean = [0.1621]
        std = [0.2751]
    elif dataset == 'promise':
        mean = [0.2565]
        std = [0.2268]
    
    if dataset in ['land', 'road', 'building']:
        train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.RandomCrop(size, size, p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
            ])
        
    elif dataset in ['chaos', 'promise']:
        train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.RandomCrop(size, size, p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
            ]) 

    test_transform = A.Compose([
        A.CenterCrop(size, size, p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
        ])
    
    return train_transform, test_transform


def get_dataloader(args):
    train_image_list, val_image_list, test_image_list, class_rgb_values = get_image_list(args)
    train_transform, test_transform = get_data_transform(args.dataset)

    if args.supernet:
        assert len(val_image_list) > 0
        train_set = TrainValDataset(args.dataset, train_image_list, val_image_list, train_transform, class_rgb_values)               
    else:
        assert len(val_image_list) == 0
        train_set = Dataset(args.dataset, train_image_list, train_transform, class_rgb_values)
    test_set = Dataset(args.dataset, test_image_list, test_transform, class_rgb_values)  
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    return train_loader, test_loader