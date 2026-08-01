#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : evaluate.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import AudioDataset
from model.model import TFMamba

CLASS_NAMES = ["Mavic2", "Mavic3", "Phantom4", "Avata", "M300"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio-Visual UAV detection and localization")
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
    parser.add_argument("--val_split_path", type=str,
                        default="/mnt/ds/MMAUD/Data-M/annotation/annotation_test_all/trainval.txt",
                        help="val split file, format: cls/file_name.npy")
    parser.add_argument("--save_path", type=str, default="output/", help="directory for evaluation figures")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--workers", type=int, default=1, help="dataloader workers")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--resume", type=str, required=True, help="checkpoint path")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "figure"), exist_ok=True)

    test_dataset = AudioDataset(
        args.val_split_path, args.gt_cls_path, args.gt_position_path,
        args.audio_path, args.img_path, args.lidar_path, args.gt_3d_label, dark_aug=1
    )
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False)

    model = TFMamba(num_cls=5, mode="test")
    model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    evaluate(model, test_loader, device, args.save_path)


def evaluate(model, valid_loader, device, save_path):
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_pos = []
    all_gt_pos = []

    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader), unit="batch"):
            spectrogram, cls_gt, pos_gt, image, img_uav_detect, img_uav_pos, _ = [
                d.to(device) for d in data
            ]
            cls_pred, pos_pred, uav_or_not, uav_pos = model(spectrogram, image)

            _, preds = torch.max(cls_pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cls_gt.cpu().numpy())
            all_pred_pos.append(pos_pred.cpu().numpy())
            all_gt_pos.append(pos_gt.cpu().numpy())

    result_pos = np.concatenate(all_pred_pos, axis=0)
    gt_pos = np.concatenate(all_gt_pos, axis=0)

    conf_matrix = accuracy(all_preds, all_labels)
    plot_confusion_matrix(conf_matrix, CLASS_NAMES, save_path)
    average_pos_err(result_pos, gt_pos)
    pos_x_y_z_err(result_pos, gt_pos)


def accuracy(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).astype(int).sum().item() / len(all_labels)
    print("Accuracy: %s" % acc)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix")
    print(conf_matrix)
    return conf_matrix


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Confusion Matrix")
    out = os.path.join(save_path, "figure", "confusion_matrix.svg")
    plt.savefig(out)
    plt.close()
    print("Saved confusion matrix to", out)


def average_pos_err(pos_pred, pos_gt):
    n = pos_pred.shape[0]
    distance = (pos_pred - pos_gt) ** 2
    ape = np.sqrt(distance[:, 0] + distance[:, 1] + distance[:, 2]).sum() / n
    print("APE: %s" % ape)


def pos_x_y_z_err(pos_pred, pos_gt):
    n = pos_pred.shape[0]
    print("Dx:", np.abs(pos_pred[:, 0] - pos_gt[:, 0]).sum() / n)
    print("Dy:", np.abs(pos_pred[:, 1] - pos_gt[:, 1]).sum() / n)
    print("Dz:", np.abs(pos_pred[:, 2] - pos_gt[:, 2]).sum() / n)


if __name__ == "__main__":
    main()
