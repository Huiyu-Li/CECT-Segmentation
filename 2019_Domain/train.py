import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import time
from visdom import Visdom
from torch.utils.data import DataLoader, random_split
# my packages
from unet3DwithClassifiy import UNet
from eval import eval_net
from utils.dataset import CustomDataset
from utils.split_data import Split_data
from utils.logger import Logger

def train_net(net, device, epochs=1, batch_size=1, lr=0.005, save_cp=True):
    exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    ckpt_dir = os.path.join(args.checkpoint + exp_id)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = os.path.join(ckpt_dir, 'log')
    sys.stdout = Logger(logfile)  # see utils.py

    train_loader = DataLoader(CustomDataset(dir_csv=os.path.join(args.csv,'train.csv')), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(CustomDataset(dir_csv=os.path.join(args.csv,'valid.csv')), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    class_num = 10
    n_train = len(train_loader)

    for epoch in range(epochs):
        net.train()
        epoch_loss1 = 0
        epoch_loss2 = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                true_class = batch['class']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_class = true_class.to(device=device, dtype=torch.long)

                masks_pred, class_pred = net(imgs)

                loss1 = criterion(masks_pred[:,0].float(), true_masks[:,0].float())
                tmp = torch.zeros(batch_size, class_num).cuda()
                label_one_hot = tmp.scatter_(1, true_class, 1)
                loss2 = criterion(class_pred.float(), label_one_hot.float())
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()

                pbar.set_postfix(**{'loss (batch)': loss1.item()})

                optimizer.zero_grad()
                loss = 0.15*loss1+0.85*loss2
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

        if save_cp:
            torch.save(net.state_dict(),os.path.join(ckpt_dir ,f'epoch{epoch + 1}.pth'))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,#15000
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,#32
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--checkpoint', dest='checkpoint', type=str, default='./checkpoints/',
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--data', dest='data', type=str, default='/media/lihuiyu/NewData/',
                        help='data set dir for split data usage')
    parser.add_argument('-c', '--csv', dest='csv', type=str, default='./GtTVTcsv/',
                        help='csv dir for data loader')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # Split_data(args.data,args.csv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=2, bilinear=False)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
