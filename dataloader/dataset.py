#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : dataset.py

import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from .data_process import audio_to_spectrogram, concat_audio, image_darkaug


class AudioDataset(Dataset):
    def __init__(self, annotation_path, gt_cls_path, gt_postion_path, audio_path, img_path,
                 lidar_path, gt_img_cls_pos_path, mode="train", dark_aug=0):
        super(AudioDataset, self).__init__()
        with open(annotation_path, "r") as f:
            self.annotation_lines = f.readlines()
        self.gt_cls_path = gt_cls_path
        self.gt_postion_path = gt_postion_path
        self.audio_path = audio_path
        self.image_path = img_path
        self.lidar_path = lidar_path
        self.mode = mode
        self.dark_aug = dark_aug
        self.gt_img_cls_pos_path = gt_img_cls_pos_path

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        file_name = self.annotation_lines[index][:-1]

        gt_cls_path = os.path.join(self.gt_cls_path, file_name)
        gt_position_path = os.path.join(self.gt_postion_path, file_name)
        uav_detect_pos_path = os.path.join(self.gt_img_cls_pos_path, file_name)

        uav_detect_pos = np.load(uav_detect_pos_path)
        uav_detect = np.array([uav_detect_pos[0]])
        uav_pos = uav_detect_pos[1:].astype(np.float64)
        uav_pos[0] = uav_pos[0] / 1280.0
        uav_pos[1] = uav_pos[1] / 720.0
        uav_pos = torch.from_numpy(uav_pos)

        image_name = os.path.join(self.image_path, file_name[:-4] + ".png")
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        scale_w = 720.0 / 256.0
        scale_h = 1280.0 / 256.0
        uav_pos[0] = uav_pos[0] / scale_w
        uav_pos[1] = uav_pos[1] / scale_h

        image, uav_detect, brightness = image_darkaug(image, uav_detect, self.dark_aug)
        uav_detect = torch.from_numpy(uav_detect)[0]
        image = np.transpose(image, [2, 0, 1])
        image = torch.from_numpy(image).float()

        lidar_name = os.path.join(self.lidar_path, file_name)
        lidar_pos = torch.from_numpy(np.load(lidar_name)).float()

        audio = concat_audio(self.audio_path, file_name)
        spectrogram = audio_to_spectrogram(audio).float()

        gt_cls = torch.tensor(np.load(gt_cls_path)[0])
        gt_position = torch.from_numpy(np.array(np.load(gt_position_path))).float()

        return spectrogram, gt_cls, gt_position, image, uav_detect, uav_pos, lidar_pos
