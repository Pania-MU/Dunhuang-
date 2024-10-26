from buddha_loader import Buddha_PathLoader, BuddhaLoader
import torch.backends.cudnn as cudnn
import numpy as np
import os
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from torchvision import transforms
from UNet import AUNet
from fcn_model import fcn_resnet50, fcn_resnet101
from networks2 import densenet201
from networks2_t import resnet50

def train_func(model_name='DAGNet', resume=False, base_lr=0.001, batch_size=2, reset_lr_epoch=5,
               train_epoch=10, folder_num=5, folder=0, buddha_data_path=''):
    #################

    # network and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  #
    print('model name : ' + model_name + '     resume : ' + str(resume)
          + '   folder / folder_num : ' + str(folder) + '/' + str(folder_num)
          + '     device : ' + device + '   batch size : ' + str(batch_size))
    if model_name == 'AUNet':
        MBC_Networks = AUNet()
    if model_name == 'FCN_R50':
        MBC_Networks = fcn_resnet50(False, num_classes=3, pretrain_backbone=resume)
    if model_name == 'FCN_R101':
        MBC_Networks = fcn_resnet101(False, num_classes=3, pretrain_backbone=resume)
    if model_name == 'densenet':
        MBC_Networks = densenet201(pretrained=resume, num_classes=21)
    if model_name == 'resnet50':
        MBC_Networks = resnet50(pretrained=resume, num_classes=21)

    MBC_Networks = MBC_Networks.to(device)
    if device == 'cuda':
        MBC_Networks = torch.nn.DataParallel(MBC_Networks)
        cudnn.benchmark = False
    optimizer = optim.SGD(MBC_Networks.parameters(), lr=base_lr, momentum=0.9)
    # loss
    CEloss = nn.CrossEntropyLoss()
    # train and validate data loader
    BPL = Buddha_PathLoader(buddha_data_path=buddha_data_path)
    train_buddha_data, val_buddha_data = BPL.Load_buddha_list(folder_num=folder_num, folder=folder)
    train_data_sampler = torch.utils.data.sampler.RandomSampler(train_buddha_data)
    BL_tra = BuddhaLoader(train_buddha_data, augmentation=False)
    train_data = torch.utils.data.DataLoader(BL_tra, batch_size=batch_size, shuffle=False,
                                             sampler=train_data_sampler, num_workers=0)
    BL_val = BuddhaLoader(val_buddha_data)
    val_data_sampler = torch.utils.data.sampler.SequentialSampler(val_buddha_data)
    val_data = torch.utils.data.DataLoader(BL_val, batch_size=batch_size, shuffle=False,
                                           sampler=val_data_sampler, num_workers=0)

    start_epoch = 0
    for name, param in MBC_Networks.named_parameters():
        param.requires_grad = True
    # train loop
    lr = base_lr
    # switch to train mode
    MBC_Networks.train()
    for epoch in range(start_epoch, start_epoch + train_epoch):
        # lower learning rate
        if (epoch + 1) % reset_lr_epoch == 0:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        running_loss = 0.0
        count = 0

        if epoch < 1000:
            t_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = t_lr
            for i, (buddha_data, label, buddha_id) in enumerate(train_data):
                optimizer.zero_grad()
                buddha_id = buddha_id[0]
                buddha_data = buddha_data.to(device).float()
                label = label.to(device)
                # segmentation networks
                buddha_h, att_map = MBC_Networks(buddha_data)
                loss = CEloss(buddha_h, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += 1

        print('epoch: ' + str(epoch) + '     loss: ' + str(running_loss / count) + '    learning rate: ' + str(lr))
        # ave_loss = running_loss / count

        # Validate
        # switch to test mode
        MBC_Networks.eval()
        total_count = 0.00001
        correct = 0

        for i, (buddha_data, label, buddha_id) in enumerate(val_data):
            buddha_id = buddha_id[0]
            buddha_data = buddha_data.to(device).float()
            label = label.to(device)
            buddha_class, _ = MBC_Networks(buddha_data)
            buddha_class = torch.softmax(buddha_class, dim=1)
            pred = buddha_class.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_in = pred.eq(label.view_as(pred)).byte()[:, 0]
            correct += correct_in.view(-1).sum().item()
            total_count += correct_in.view(-1).size()[0]

        print('     Test correct/test count: %s/%s AVE Precision: %.3f ' %
              (correct, total_count, 100. * correct / total_count))

    return total_count, correct

