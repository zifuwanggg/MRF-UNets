from torch.utils.data import DataLoader

from datas.dataset import Dataset
from datas.train_val_dataset import TrainValDataset
from datas.data_utils import get_image_list, get_data_transform


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