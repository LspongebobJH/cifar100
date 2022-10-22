# train.py
# In this branch, any modifications will be DISCARDED!!!

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
from utils import fix_all_params, get_training_dataloader, get_test_dataloader, WarmUpLR, get_network

def train(epoch):

    start = time.time()
    backbone.train()
    ft.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        images, idx = aux_transforms(images, trans_list=['crop', 'horflip'], replica=args.replica)
        idx = idx.to(args.device)
        target_imgs_idx = [th.where(idx == i)[0][0].item() for i in th.unique(idx)]
        labels = labels.to(args.device)
        images = images.to(args.device)

        optimizer.zero_grad()
        emb = backbone(images)
        if args.aggr_type == 'attn':
            g = build_g(emb, idx).to(args.device)
            emb = aggr(g, g.ndata['feat'], attn=False)[target_imgs_idx]
        elif args.aggr_type == 'ws':
            emb = aggr(emb, idx)
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


        # if epoch <= args.warm:
        #     warmup_scheduler.step()

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
        images, idx = aux_transforms(images, trans_list=['crop', 'horflip'], replica=2)
        idx = idx.to(args.device)
        target_imgs_idx = [th.where(idx == i)[0][0].item() for i in th.unique(idx)]
        labels = labels.to(args.device)
        images = images.to(args.device)

        emb = backbone(images)
        if args.aggr_type == 'attn':
            g = build_g(emb, idx).to(args.device)
            emb = aggr(g, g.ndata['feat'], attn=False)[target_imgs_idx]
        elif args.aggr_type == 'ws':
            emb = aggr(emb, idx)
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
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/resnet50_base/checkpoints/')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--ft-type', type=str, default='base', choices=['base'])
    parser.add_argument('--aggr-type', type=str, default='attn', choices=['attn', 'ws'])
    parser.add_argument('--replica', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--base', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate') 
    parser.add_argument('--save-epochs', type=int, default=1)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)

    # attn
    parser.add_argument('--hid-dim', type=int, default=4)
    parser.add_argument('--n-heads-list', nargs='+', type=int, default=[1])

    # training
    parser.add_argument('--milestones', nargs='+', type=int, default=[1, 3, 5])

    # deprecate
    # parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    
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
    num_classes = backbone.fc.out_features
    backbone.fc = nn.Identity()
    checkpoint = th.load(f'{args.checkpoint_path}{args.backbone}-final-best.pth')
    backbone.load_state_dict(checkpoint)
    fix_all_params(backbone)
    backbone = backbone.to(args.device)

    # create ft part
    if args.ft_type == 'base':
        ft = nn.Linear(in_dim, num_classes)
    checkpoint = th.load(f'{args.checkpoint_path}{args.ft_type}-final-best.pth')
    ft.load_state_dict(checkpoint)
    fix_all_params(ft)
    ft = ft.to(args.device)

    # create aggr part
    if args.aggr_type == 'attn':
        aggr = GAT(n_layers=1, n_heads_list=args.n_heads_list, in_dim=in_dim, 
                hid_dim=args.hid_dim, out_dim=in_dim, dropout=0.6, neg_slope=0.2)
    elif args.aggr_type == 'ws':
        aggr = WS(n_in_nodes=5, in_dim=in_dim, out_dim=in_dim)
    aggr = aggr.to(args.device)

    # set components
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(aggr.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.milestones == [-1]: # no train scheduer
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.2)
    elif len(args.milestones) == 1:
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.milestones[0], gamma=0.2)
    else:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # create checkpoint folder to save model
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.aggr_type, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{aggr}-{epoch}-{type}.pth')

    best_acc = 0.0

    # start training
    for epoch in range(1, args.epochs + 1):
        # if epoch > args.warm:
        # train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
        train_scheduler.step()

        #start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:
            weights_path = checkpoint_path.format(aggr=args.aggr_type, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            th.save(backbone.state_dict(), weights_path)

            best_acc = acc
            continue

        if not epoch % args.save_epochs:
            weights_path = checkpoint_path.format(aggr=args.aggr_type, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            th.save(backbone.state_dict(), weights_path)