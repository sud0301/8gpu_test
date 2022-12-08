import argparse
import numpy as np
import sys
import os
import os.path as osp
#import scipy.misc
import timeit

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform

from model.deeplabv3p import DeepV3PlusW38 as Res_Deeplab
from data import get_loader, get_data_path

start = timeit.default_timer()

CHECKPOINT_DIR = './checkpoints/'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input-size", type=str, default='544,544',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--ignore-label", type=float, default=250,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=20000,
                        help="Number of iterations")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of iterations")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=1000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    label = label.long().cuda(gpu)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # create network
    model = Res_Deeplab(num_classes= args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    cudnn.enabled = True
    cudnn.benchmark = True

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    data_loader = get_loader('cityscapes')
    data_path = get_data_path('cityscapes')
    train_dataset = data_loader( data_path, split='train', img_size=[input_size[0], input_size[1]])    
    trainloader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
    trainloader_iter = iter(trainloader)
 
    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.parameters(),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # loss/ bilinear upsampling
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    args.num_steps = int(args.num_epochs*len(trainloader))

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        try:
            batch_lab = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch_lab = next(trainloader_iter)

        images, labels = batch_lab
        images = images.cuda()
        pred = interp(model(images))
        loss = loss_calc(pred, labels, 0)
        loss_value = loss.item()
        
        loss.backward()
        optimizer.step()

        if i_iter%10==0:
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(i_iter, args.num_steps, loss_value))

        if i_iter == args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint ...')
            torch.save(model.state_dict(),osp.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
