# Audio-Visual Cross Mamba for UAV Detection & Localization

Multimodal model that fuses **audio spectrograms** and **RGB images** with Mamba backbones and cross-attention, for UAV type classification and 3D position estimation.

## Features

- Time / frequency Mamba branches on multi-channel audio mel-spectrograms
- Image Mamba branch with auxiliary detect / 2D position heads
- Cross-attention fusion (audio–audio, then audio–visual)
- Multi-task training: UAV class + 3D position + image detect + image 2D position

## Repository structure

```
.
├── train.py                 # training entry
├── evaluate.py              # evaluation entry (acc / APE / confusion matrix)
├── model/
│   ├── model.py             # TFMamba
│   ├── mamba.py             # Mamba / ImgMamba / selective scan
│   └── attention.py         # ConvergeAttention
├── dataloader/
│   ├── dataset.py           # AudioDataset
│   └── data_process.py      # mel spectrogram & image aug
├── kernels/selective_scan/  # CUDA selective-scan extension
├── requirements.txt
└── output/                  # checkpoints (created at runtime)
```

## Environment

Tested with conda env (`Python 3.10`):

| Package        | Version   |
|----------------|-----------|
| PyTorch        | 2.1.1     |
| torchvision    | 0.16.1    |
| torchaudio     | 2.1.1     |
| mamba-ssm      | 1.1.1     |
| CUDA GPU       | required for selective scan |

### Install

```bash
conda create -n mb python=3.10 -y
conda activate mb

# Install PyTorch matching your CUDA (example: cu118)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# Build selective_scan CUDA kernel (required)
cd kernels/selective_scan
pip install .
cd ../..
```

Verify:

```bash
python -c "import torch, mamba_ssm, selective_scan_cuda; print(torch.__version__, torch.cuda.is_available())"
```

## Dataset layout

Expect MMAUD-style paths (override via CLI):

```
Data-M/
├── audio_npy/          # *.npy audio, shape [T, 4]
├── image/              # *.png RGB frames
├── lidar/              # *.npy lidar / supervised 3D position
├── label/              # *.npy class id
├── gt/                 # *.npy 3D position gt
├── img_3d_label/       # *.npy [detect, u, v]
└── annotation/
    ├── annotation_train_all/trainval.txt
    └── annotation_test_all/trainval.txt
```

Each split line looks like: `0/111.npy` (`class_folder/file.npy`).

## Train

```bash
python train.py \
  --audio_path /path/to/Data-M/audio_npy/ \
  --img_path /path/to/Data-M/image/ \
  --lidar_path /path/to/Data-M/lidar/ \
  --gt_cls_path /path/to/Data-M/label/ \
  --gt_position_path /path/to/Data-M/gt/ \
  --gt_3d_label /path/to/Data-M/img_3d_label/ \
  --train_split_path /path/to/Data-M/annotation/annotation_train_all/trainval.txt \
  --val_split_path /path/to/Data-M/annotation/annotation_test_all/trainval.txt \
  --save_path output/ \
  --batch_size 64 \
  --train_epoch 200 \
  --gpu cuda:0
```

Resume:

```bash
python train.py --resume output/best_model_xxx.pth ...
```

Checkpoints are written to `--save_path` as `best_model_{epoch}_{val_loss}.pth` and per-epoch files.

## Evaluate

```bash
python evaluate.py \
  --resume output/best_model_xxx.pth \
  --audio_path /path/to/Data-M/audio_npy/ \
  --img_path /path/to/Data-M/image/ \
  --lidar_path /path/to/Data-M/lidar/ \
  --gt_cls_path /path/to/Data-M/label/ \
  --gt_position_path /path/to/Data-M/gt/ \
  --gt_3d_label /path/to/Data-M/img_3d_label/ \
  --val_split_path /path/to/Data-M/annotation/annotation_test_all/trainval.txt \
  --save_path output/ \
  --gpu cuda:0
```

Reports:

- Classification accuracy and confusion matrix (saved to `output/figure/confusion_matrix.svg`)
- Average Position Error (APE)
- Per-axis errors (Dx / Dy / Dz)

Classes: `Mavic2`, `Mavic3`, `Phantom4`, `Avata`, `M300`.

## Model I/O

| Input | Shape |
|-------|--------|
| Spectrogram | `[B, 4, 224, 16]` |
| Image | `[B, 3, 256, 256]` |

| Output | Shape |
|--------|--------|
| Class logits | `[B, 5]` |
| 3D position | `[B, 3]` |
| Image detect logits | `[B, 2]` |
| Image 2D position | `[B, 2]` |

## Cite
```
@ARTICLE{11479681,
author={Xiao, Zhenyuan and Yuan, Shenghai and Xu, Guili and Zeng, Xianglong and Hu, Huanran and He, Junwei and Yang, Yizhuo},
journal={IEEE Sensors Journal}, 
title={AV-DTEC: Self-Supervised Audio–Visual Fusion for Drone 3-D Trajectory Estimation and Classification}, 
year={2026},
volume={26},
number={10},
pages={15912-15924},
doi={10.1109/JSEN.2026.3680898}}
```

## License

Code is provided for research use. Please cite / credit the authors if you use this repository.
