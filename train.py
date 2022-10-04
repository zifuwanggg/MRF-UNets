import os
import time
import logging
import argparse
from glob import glob

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from utils import initialize
from metric.iou_dice import IoUDice
from jaccard_loss import JaccardLoss
from models.mrf_unet import ChildNet
from datas.dataloader import get_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', dest='output', type=str, default="../outputs/train")
    parser.add_argument('--data-dir', dest='data_dir', type=str, default="/Users/whoami/datasets")
    parser.add_argument('--dataset', dest='dataset', type=str, default="land")
    parser.add_argument('--freq', type=int, default=10, dest='freq')
    parser.add_argument('--workers', type=int, default=4, dest='workers')
    parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    parser.add_argument('--size', type=int, default=256, dest='size')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs')
    parser.add_argument('--learning-rate-weights', type=float, default=0.0005, dest='lr_weights')
    parser.add_argument('--weight-decay-weights', type=float, default=0.0001, dest='weight_decay_weights')
    parser.add_argument('--channel-step', type=int, default=5, dest='channel_step')
    parser.add_argument('--choices', dest='choices', type=str, default="8,8,3,3,1,3,3,1,3,3,1,3,3,1,0,8,1,0,8,1,0,8,1,0,8,1")
    args = parser.parse_args()
    
    args.supernet = False
    
    if args.dataset == 'land':
        args.image_channels = 3
        args.num_classes = 6
        args.ignore_index = 6
    elif args.dataset in ['road', 'building']:
        args.image_channels = 3
        args.num_classes = 1
        args.ignore_index = None
    elif args.dataset == 'chaos':
        args.image_channels = 1
        args.num_classes = 5
        args.ignore_index = None
    elif args.dataset == 'promise':
        args.image_channels = 1
        args.num_classes = 1
        args.ignore_index = None
    
    return args


def main(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    train_loader, test_loader = get_dataloader(args)

    choices = np.array([int(c) for c in args.choices.split(',')])
    model = ChildNet(args.image_channels, args.num_classes, args.channel_step, choices).to(device)
    initialize(model)
    
    optimizer_weights = optim.Adam(model.parameters(), lr=args.lr_weights, weight_decay=args.weight_decay_weights)
    
    lr_lambda = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_weights, lr_lambda=lr_lambda)
   
    criterion = JaccardLoss(ignore_index=args.ignore_index).to(device)
    
    start_epoch = 0
    
    ckp_dir = os.path.join(args.output, 'checkpoints')    
    os.makedirs(ckp_dir, exist_ok=True)  
    avail = glob(os.path.join(ckp_dir, 'checkpoint*.pth'))
    avail = [(int(f[-len('.pth') - 3:-len('.pth')]), f) for f in avail]
    avail = sorted(avail)
    ckp_path = avail[-1][1] if avail else None

    if ckp_path and os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=device)
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer_weights.load_state_dict(checkpoint['optimizer_weights'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info(f"Checkpoint {ckp_path} is loaded")
    else:
        logging.info(f"No checkpoint is found") 
        
    for epoch in range(start_epoch, args.epochs):
        train(model, train_loader, optimizer_weights, criterion, device, epoch, args)
        
        if epoch == args.epochs - 1:
            test(model, test_loader, device, epoch, args)

        scheduler.step()    
            
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer_weights': optimizer_weights.state_dict(),
                 'lr_scheduler': scheduler.state_dict()}
        
        filename = f"checkpoint{epoch+1:03d}.pth"
        filename = os.path.join(ckp_dir, filename)
        torch.save(state, filename)


def train(model, train_loader, optimizer_weights, criterion, device, epoch, args):  
    logging.info(f"*****Begin train epoch {epoch + 1}*****")
    
    model.train()
    
    end = time.time()
    for iter, (image, label) in enumerate(train_loader):
        start = time.time()
        toprint = f"[{epoch + 1}][{iter}|{len(train_loader)}], data time: {(start - end):.6f}, "

        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)

        optimizer_weights.zero_grad()

        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        
        optimizer_weights.step()
            
        end = time.time()  
        toprint += f"batch time: {(end - start):.6f}, "

        if iter % args.freq == 0:
            lr = optimizer_weights.param_groups[0]['lr']
            toprint += f"learning rate: {lr:.6f}, loss: {loss.item():.6f}"
            logging.info(toprint)

    logging.info(f"*****Finish train epoch {epoch + 1}*****\n")
    
  
def test(model, test_loader, device, epoch, args):
    logging.info(f"*****Begin test epoch {epoch + 1}*****")
    
    model.eval()
    
    iou_dice = IoUDice(args.num_classes, device, args.dataset, args.ignore_index)

    end = time.time()
    for iter, (image, label) in enumerate(test_loader):
        start = time.time()
        toprint = f"[{epoch + 1}][{iter}|{len(test_loader)}], data time: {(start - end):.6f}, "
        
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)

        with torch.no_grad():
            logits = model(image)
            
            if args.num_classes > 1:
                prob = logits.log_softmax(dim=1).exp()
                pred = prob.argmax(1)
            else:
                prob = F.logsigmoid(logits).exp()
                pred = (prob > 0.5).squeeze(1).to(torch.long)
                
            iou_dice.add(pred, label)
        
        iou, dice = iou_dice.value()    
        
        end = time.time()  
        toprint += f"batch time: {(end - start):.6f}, "
        
        if iter % args.freq == 0: 
            toprint += f"IoU: {iou:.2f}, Dice: {dice:.2f}"
            logging.info(toprint)
    
    iou, dice = iou_dice.value()
    
    logging.info(f"*****Finish test epoch {epoch + 1}*****\n")
    logging.info(f"IoU: {iou:.2f}, Dice: {dice:.2f}")


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output, exist_ok=True)

    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
 
    logging.basicConfig(filename=os.path.join(args.output, "train.log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(str(args))

    main(args)