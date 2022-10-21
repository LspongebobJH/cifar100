# train.py

import os
import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from gnn import *
from gnn_utils import *

from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, get_network

def train(epoch):

    start = time.time()
    backbone.train()
    ft.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if not args.base:
            images, idx = aux_transforms(images, trans_list=['crop', 'horflip'], replica=2)
            idx = idx.to(args.device)
            target_imgs_idx = [th.where(idx == i)[0][0].item() for i in th.unique(idx)]
        labels = labels.to(args.device)
        images = images.to(args.device)

        optimizer.zero_grad()
        emb = backbone(images)
        if not args.base:
            if args.ft_type == 'attn':
                g = build_g(emb, idx).to(args.device)
                logits = ft(g, g.ndata['feat'], attn=False)[target_imgs_idx]
            elif args.ft_type == 'ws':
                logits = ft(emb, idx)
        else:
            logits = ft(emb)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))


        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@th.no_grad()
def eval_training(epoch=0):

    start = time.time()
    backbone.eval()
    ft.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        if not args.base:
            images, idx = aux_transforms(images, trans_list=['crop', 'horflip'], replica=2)
            idx = idx.to(args.device)
            target_imgs_idx = [th.where(idx == i)[0][0].item() for i in th.unique(idx)]
        labels = labels.to(args.device)
        images = images.to(args.device)

        emb = backbone(images)
        if not args.base:
            if args.ft_type == 'attn':
                g = build_g(emb, idx).to(args.device)
                logits = ft(g, g.ndata['feat'], attn=False)[target_imgs_idx]
            elif args.ft_type == 'ws':
                logits = ft(emb, idx)
        else:
            logits = ft(emb)
        loss = loss_function(logits, labels)

        test_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--ft-type', type=str, default='attn', choices=['attn', 'ws'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--base', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate') 
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)

    # attn
    parser.add_argument('--hid-dim', type=int, default=4)
    parser.add_argument('--n-heads-list', nargs='+', type=int, default=[1])
    
    args = parser.parse_args()
    print(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )

     # create backbone
    backbone = get_network(args.backbone)
    in_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(args.device)

    # Create attention aggregation part
    if not args.base:
        if args.ft_type == 'attn':
            ft = GAT(n_layers=1, n_heads_list=args.n_heads_list, in_dim=in_dim, 
                    hid_dim=args.hid_dim, out_dim=args.num_classes, dropout=0.6, neg_slope=0.2).to(args.device)
        elif args.ft_type == 'ws':
            ft = WS(n_in_nodes=5, in_dim=in_dim, out_dim=args.num_classes).to(args.device)
    else:
        ft = nn.Linear(in_dim, args.num_classes).to(args.device)


    # set components
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(backbone.parameters())+list(ft.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.base:
        model_type = 'base'
    else:
        model_type = args.ft_type
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.backbone+'_'+model_type, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{model}-{epoch}-{type}.pth')

    best_acc = 0.0

    # start training
    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(model=args.backbone, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            th.save(backbone.state_dict(), weights_path)
            
            weights_path = checkpoint_path.format(model=args.ft_type, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            th.save(ft.state_dict(), weights_path)

            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(model=args.backbone, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            th.save(backbone.state_dict(), weights_path)

            weights_path = checkpoint_path.format(model=args.ft_type, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            th.save(ft.state_dict(), weights_path)