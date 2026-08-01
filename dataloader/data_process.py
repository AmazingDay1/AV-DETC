#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : data_process.py

import os
import random

import cv2
import numpy as np
import torch
import torchaudio.transforms as audio_trans
from torchvision import transforms


def concat_audio(audio_path, file_name):
    return np.load(os.path.join(audio_path, file_name))


def audio_to_spectrogram(audio, sr=48000, spectrogram_process_mode=1):
    mel_spectrogram = audio_trans.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=1024,
        n_mels=200,
        pad_mode="constant",
        norm="slaney",
        mel_scale="slaney",
        power=2,
    )
    audio_data = torch.tensor(audio, dtype=torch.float32)
    spectrogram = mel_spectrogram(audio_data)

    if spectrogram_process_mode == 1:
        spectrogram = scale_processing(spectrogram)
    else:
        spectrogram = normalization_processing(spectrogram)

    transform = transforms.Resize((224, 16), antialias=True)
    spectrogram = transform(spectrogram)
    return spectrogram  # [4, 224, 16]


def scale_processing(data):
    for i in range(data.shape[0]):
        data_min = torch.min(data[i, :])
        data_max = torch.max(data[i, :])
        data[i, :] = (data[i, :] - data_min) / (data_max - data_min)
    return data


def normalization_processing(data):
    data_m = torch.mean(data)
    data_s = torch.std(data)
    return (data - data_m) / data_s


def img_brightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    return cv2.addWeighted(img1, c, blank, 1 - c, b)


def image_darkaug(img, img_label, dark_aug, brightness=1):
    if dark_aug == 1:
        if random.random() > 0.6:
            brightness = 0
            img = img_brightness(img, brightness, 3)
            img_label = np.array([0])
    elif dark_aug == 2:
        img = img_brightness(img, brightness, 3)
        img_label = np.array([0])
    elif dark_aug > 2:
        brightness = 0.04 / dark_aug
        img = img_brightness(img, brightness, 3)
        img_label = np.array([0])

    img = preprocess_input(img)
    return img, img_label, brightness


def preprocess_input(image):
    return image / 127.5 - 1
