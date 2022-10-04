import os
import time
import random
import logging
import argparse
from glob import glob

import torch

from utils import initialize
from jaccard_loss import JaccardLoss
from models.mrf_unet import MRFSuperNet
from datas.dataloader import get_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='Search', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--output', dest='output', type=str, default="../outputs/search")
    parser.add_argument('--data-dir', dest='data_dir', type=str, default="/Users/whoami/datasets")
    parser.add_argument('--freq', type=int, default=10, dest='freq')
    parser.add_argument('--workers', type=int, default=4, dest='workers')
    parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    parser.add_argument('--size', type=int, default=256, dest='size')
    parser.add_argument('--epochs', type=int, default=50, dest='epochs')
    parser.add_argument('--learning-rate-weights', type=float, default=0.0005, dest='lr_weights')
    parser.add_argument('--learning-rate-potentials', type=float, default=0.0003, dest='lr_potentials')
    parser.add_argument('--weight-decay-weights', type=float, default=0.0001, dest='weight_decay_weights')
    parser.add_argument('--weight-decay-potentials', type=float, default=0.0001, dest='weight_decay_potentials')
    parser.add_argument('--warmup', type=int, default=10, dest='warmup')
    parser.add_argument('--long', type=int, default=10000, dest='long')
    parser.add_argument('--short', type=int, default=10, dest='short')
    parser.add_argument('--tau', type=float, default=1., dest='tau')
    parser.add_argument('--channel-step', type=int, default=5, dest='channel_step')
    args = parser.parse_args()
    
    args.supernet = True
    args.dataset = 'land'
    args.image_channels = 3
    args.num_classes = 6
    args.ignore_index = 6
    
    return args


def main(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    train_val_loader, _ = get_dataloader(args)

    model = MRFSuperNet(image_channels=args.image_channels, num_classes=args.num_classes, channel_step=args.channel_step).to(device)
    initialize(model)
    
    optimizer_weights = torch.optim.Adam(model.weights(), lr=args.lr_weights, weight_decay=args.weight_decay_weights)
    optimizer_potentials = torch.optim.Adam(model.potentials(), lr=args.lr_potentials, weight_decay=args.weight_decay_potentials)
    lr_lambda = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_weights, lr_lambda)
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
        optimizer_potentials.load_state_dict(checkpoint['optimizer_potentials'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info(f"Checkpoint {ckp_path} is loaded")
    else:
        logging.info(f"No checkpoint is found") 

    for epoch in range(start_epoch, args.epochs):
        train(model, train_val_loader, optimizer_weights, optimizer_potentials, criterion, device, epoch, args)

        scheduler.step()    
            
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer_weights': optimizer_weights.state_dict(),
                 'optimizer_potentials': optimizer_potentials.state_dict(),
                 'lr_scheduler': scheduler.state_dict()}
        
        filename = f"checkpoint{epoch+1:03d}.pth"
        filename = os.path.join(ckp_dir, filename)
        torch.save(state, filename)
        

def train(model, train_val_loader, optimizer_weights, optimizer_potentials, criterion, device, epoch, args):  
    logging.info(f"*****Begin train epoch {epoch+1}*****")
    
    model.train()
    
    factor_prod = model.create_factor_prod()
    choices = model.burnin(factor_prod, args.long, choices=None)

    end = time.time()
    for iter, (train_image, train_label, val_image, val_label) in enumerate(train_val_loader):
        start = time.time()
        toprint = f"[{epoch + 1}][{iter}|{len(train_val_loader)}], data time: {(start - end):.6f}, "

        train_image = train_image.to(device=device, dtype=torch.float32)
        train_label = train_label.to(device=device, dtype=torch.long)

        optimizer_weights.zero_grad()
        
        if random.random() < 0.5:
            kernel_size = 3
        else:
            kernel_size = 5

        choices_one_hot_max = model.get_choices_one_hot(mode=f"max{kernel_size}").to(device)
        train_logits_max = model(train_image, choices_one_hot_max)
        train_loss_max = criterion(train_logits_max, train_label)
        train_loss_max.backward()
        
        choices_one_hot = model.get_choices_one_hot().to(device)
        train_logits = model(train_image, choices_one_hot)
        train_loss = criterion(train_logits, train_label)
        train_loss.backward()
                
        optimizer_weights.step()
        
        val_loss = torch.zeros(1).to(device)
        
        if epoch >= args.warmup:
            val_image = val_image.to(device=device, dtype=torch.float32)
            val_label = val_label.to(device=device, dtype=torch.long)

            optimizer_potentials.zero_grad()

            factor_prod = model.create_factor_prod()
            choices = model.burnin(factor_prod, args.short, choices)
            choices_one_hot = model.sample(factor_prod, args.tau, choices).to(device)
            val_logits = model(val_image, choices_one_hot)
            val_loss = criterion(val_logits, val_label)
            val_loss.backward()
        
            optimizer_potentials.step()

        end = time.time()  
        toprint += f"batch time: {(end - start):.6f}, "
        
        if iter % args.freq == 0:
            lr = optimizer_weights.param_groups[0]['lr']
            toprint += f"learning rate: {lr:.6f}, train_loss: {train_loss.item():.6f}, val_loss: {val_loss.item():.6f}"
            logging.info(toprint)
        
    logging.info(f"*****Finish train epoch {epoch + 1}*****\n")

                    
if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output, exist_ok=True)

    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
 
    logging.basicConfig(filename=os.path.join(args.output, "search.log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(str(args))

    main(args)