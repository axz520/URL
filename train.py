import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler

from loss import CELoss, OrdinaryReg, SupConLoss

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../Datasets/LIDC_cls', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=5, type=int)

    parser.add_argument('--dim', default=1, type=int)
    parser.add_argument('--gpu_id', nargs='+', type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--single_type', default='vote', type=str)
    parser.add_argument('--point_path', default=None, type=str)
    parser.add_argument('--save_log', default='acc.log', type=str)
    parser.add_argument('--loss_type', default='celoss', type=str)
    return parser.parse_args()

def get_loader():
    pass

def get_prototype(device, feature_size, pre_target, pre_feature, num_class=5):
    pre_feature = pre_feature.float()
    prototype, count = torch.zeros((num_class, feature_size)), torch.zeros(num_class)
    
    for i in range(num_class):
        count[i] = torch.sum(pre_target == i)
        prototype[i] = torch.sum(
            torch.where((pre_target == i).reshape(-1, 1).repeat(1, pre_feature.shape[1]), pre_feature, torch.zeros(pre_feature.shape).to(device)), dim=0
        )
    for i in range(num_classes):
        prototype[i] = prototype[i] / count[i] if count[i] > 0 else prototype[i]
    
    return prototype.to(device), count.to(device)

def get_pseudo_label(prototype, features, device):
    features = features / torch.norm(features, dim=-1, keepdim=True)
    prototype = prototype / torch.max(torch.norm(prototype, dim=-1, keepdim=True), torch.full((5, 1), 1e-6).to(device))
    pseudo_label = torch.mm(features, prototype.t())
    return pseudo_label

if __name__ == '__main__':
    # param
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = get_parser()
    batch_size  = parser.batch_size
    in_channels = parser.in_channels
    input_size  = parser.input_size
    num_classes = parser.num_classes
    num_workers = parser.num_workers
    epochs      = parser.epochs
    lr          = parser.lr
    path        = parser.path
    save_log    = parser.save_log
    dim         = parser.dim
    seed        = parser.seed
    single_type = parser.single_type
    point_path  = parser.point_path
    loss_type   = parser.loss_type

    model = your_net() # your model
    feature_size = model._fc.in_features

    epoch_contrast = 10

    # clean training set and full training set
    train_clean_loader, train_loader, valid_loader, test_loader = get_loader()
    
    # loss
    criterion = CELoss().to(device)
    ordinary_penalty = OrdinaryReg().to(device)
    sup_con_Loss = SupConLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scaler = GradScaler()

    for i in range(epoch_contrast):
        for index, img, label in train_clean_loader:
            img, label = img.to(device), label.to(device)
            with autocast():
                out = model(img)
                # fc feature
                features_grad = model.batch_feature.to(device)
                features_grad = torch.reshape(features_grad, (-1, feature_size))
                features_grad = features_grad / torch.norm(features_grad, dim=-1, keepdim=True)
                loss =  sup_con_Loss(features_grad, label.argmax(1))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    p_label = torch.zeros((len(train_loader.train_dataset), num_classes)).to(device)
    a, b = 3., 1.0
    t, k = 0.1, 0.8
    for epoch in range(epochs):
        for index, img, label in train_loader:
            loss = 0.
            loss1, loss2, loss3, loss4 = 0., 0., 0., 0.
            
            img, label = img.to(device), label.to(device)
            with autocast():
                out = model(img)
                # celoss
                loss1 = criterion(out, label)
                loss += loss1
                out_softmax = torch.clamp(torch.softmax(out, dim=1), min=1e-4, max=1.0 - 1e-4)

                # batch prototype regularization
                features_grad = model.batch_feature.to(device)
                features_grad = torch.reshape(features_grad, (-1, feature_size))
                features = features_grad.detach().clone()

                prototype, count = get_prototype(device, feature_size, torch.argmax(out, dim=1), features, num_class=num_classes)
                pseudo_label = (get_pseudo_label(prototype, features, device)).to(device)
                pseudo_label = torch.softmax(pseudo_label / t, 1)

                p_label[index] = k*p_label[index] + (1-k)*pseudo_label
                loss2 = ((1 - p_label[index] * out_softmax).log().sum(dim=1)).mean()
                loss += a * loss2

                # ordinal penalty
                loss3 = ordinary_penalty(out_softmax, torch.argmax(label, dim=1))
                loss += b * loss3

                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
