#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : train.py

import argparse
import math
import os
from functools import partial

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import AudioDataset
from model.model import TFMamba


def main():
    parser = argparse.ArgumentParser(description="Audio-Visual UAV detection and 3D position estimation")
    parser.add_argument("--audio_path", type=str, default="/mnt/ds/MMAUD/Data-M/audio_npy/",
                        help="path to audio npy files")
    parser.add_argument("--img_path", type=str, default="/mnt/ds/MMAUD/Data-M/image/",
                        help="path to images")
    parser.add_argument("--lidar_path", type=str, default="/mnt/ds/MMAUD/Data-M/lidar/",
                        help="path to lidar npy files")
    parser.add_argument("--gt_cls_path", type=str, default="/mnt/ds/MMAUD/Data-M/label/",
                        help="path to classification labels")
    parser.add_argument("--gt_position_path", type=str, default="/mnt/ds/MMAUD/Data-M/gt/",
                        help="path to 3D position labels")
    parser.add_argument("--gt_3d_label", type=str, default="/mnt/ds/MMAUD/Data-M/img_3d_label/",
                        help="path to image 2D detect/position labels")
    parser.add_argument("--train_split_path", type=str,
                        default="/mnt/ds/MMAUD/Data-M/annotation/annotation_train_all/trainval.txt",
                        help="train split file, format: cls/file_name.npy")
    parser.add_argument("--val_split_path", type=str,
                        default="/mnt/ds/MMAUD/Data-M/annotation/annotation_test_all/trainval.txt",
                        help="val split file, format: cls/file_name.npy")
    parser.add_argument("--save_path", type=str, default="output/", help="directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--train_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--workers", type=int, default=1, help="dataloader workers")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    train_dataset = AudioDataset(
        args.train_split_path, args.gt_cls_path, args.gt_position_path,
        args.audio_path, args.img_path, args.lidar_path, args.gt_3d_label, dark_aug=1
    )
    test_dataset = AudioDataset(
        args.val_split_path, args.gt_cls_path, args.gt_position_path,
        args.audio_path, args.img_path, args.lidar_path, args.gt_3d_label, dark_aug=1
    )
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=True)

    model = TFMamba(num_cls=5, mode="train")
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("device:", device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    lr_fun = lr_adjust_fun("decay", 0.0001, 0.0001 * 0.01, args.train_epoch)

    pos_criterion = torch.nn.L1Loss()
    cls_criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(args.train_epoch):
        set_optim_lr(optimizer, lr_fun, epoch)
        print("current lr:", optimizer.param_groups[0]["lr"])

        train_loss = train_mode(model, train_loader, optimizer, pos_criterion, cls_criterion, device)
        valid_loss = valid_mode(model, test_loader, pos_criterion, cls_criterion, device)
        print("Epoch {}/{}, Train Loss: {}, Val Loss: {}".format(
            epoch + 1, args.train_epoch, train_loss, valid_loss
        ))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, "best_model_{}_{}.pth".format(epoch + 1, valid_loss))
            )

        torch.save(
            model.state_dict(),
            os.path.join(args.save_path, "epoch{}_val_loss_{}.pth".format(epoch + 1, valid_loss))
        )


def train_mode(model, train_loader, optimizer, pos_criterion, cls_criterion, device):
    model.train()
    train_loss = 0.0
    for data in tqdm(train_loader, total=len(train_loader), unit="batch"):
        spectrogram, cls_gt, _, image, img_uav_detect, img_uav_pos, lidar_pos = [
            d.to(device) for d in data
        ]
        optimizer.zero_grad()
        cls_pred, pos_pred, uav_or_not, uav_pos = model(spectrogram, image)

        loss_cls = cls_criterion(cls_pred, cls_gt)
        loss_pos = pos_criterion(pos_pred, lidar_pos)
        loss_uav_detect = cls_criterion(uav_or_not, img_uav_detect)
        loss_uav_pos = pos_criterion(uav_pos, img_uav_pos)

        total_loss = loss_cls + 2 * loss_pos + loss_uav_pos * 0.5 + loss_uav_detect * 0.5
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
    return train_loss / len(train_loader)


def set_optim_lr(optim, lr_adjust_fun, epoch):
    lr = lr_adjust_fun(epoch)
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def lr_adjust_fun(lr_decay_type, lr, min_lr, total_iters,
                  warmup_iters_ratio=0.05,
                  warmup_lr_ratio=0.1,
                  no_aug_iter_ratio=0.05,
                  step_num=10):

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(
                    math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        n = iters // step_size
        return lr * decay_rate ** n

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        fun = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        fun = partial(step_lr, lr, decay_rate, step_size)
    return fun


def valid_mode(model, valid_loader, pos_criterion, cls_criterion, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader), unit="batch"):
            spectrogram, cls_gt, pos_gt, image, img_uav_detect, img_uav_pos, _ = [
                d.to(device) for d in data
            ]
            cls_pred, pos_pred, uav_or_not, uav_pos = model(spectrogram, image)
            loss_cls = cls_criterion(cls_pred, cls_gt)
            loss_pos = pos_criterion(pos_pred, pos_gt)
            valid_loss += (loss_cls + loss_pos).item()
    return valid_loss / len(valid_loader)


if __name__ == "__main__":
    main()
