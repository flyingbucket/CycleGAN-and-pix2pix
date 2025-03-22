# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/bkkaggle/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] colab_type="text" id="7wNjDKdQy35h"
# # Install

# + colab={} colab_type="code" id="TRm-USlsHgEV"
# !git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# + colab={} colab_type="code" id="Pt3igws3eiVp"
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')

# + colab={} colab_type="code" id="z1EySlOXwwoa"
# !pip install -r requirements.txt

# + [markdown] colab_type="text" id="8daqlgVhw29P"
# # Datasets
#
# Download one of the official datasets with:
#
# -   `bash ./datasets/download_pix2pix_dataset.sh [cityscapes, night2day, edges2handbags, edges2shoes, facades, maps]`
#
# Or use your own dataset by creating the appropriate folders and adding in the images. Follow the instructions [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md#pix2pix-datasets).

# + colab={} colab_type="code" id="vrdOettJxaCc"
# !bash ./datasets/download_pix2pix_dataset.sh facades

# + [markdown] colab_type="text" id="gdUz4116xhpm"
# # Pretrained models
#
# Download one of the official pretrained models with:
#
# -   `bash ./scripts/download_pix2pix_model.sh [edges2shoes, sat2map, map2sat, facades_label2photo, and day2night]`
#
# Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

# + colab={} colab_type="code" id="GC2DEP4M0OsS"
# !bash ./scripts/download_pix2pix_model.sh facades_label2photo

# + [markdown] colab_type="text" id="yFw1kDQBx3LN"
# # Training
#
# -   `python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA`
#
# Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. Add `--direction BtoA` if you want to train a model to transfrom from class B to A.

# + colab={} colab_type="code" id="0sp7TCT2x9dB"
# !python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --display_id -1

# + [markdown] colab_type="text" id="9UkcaFZiyASl"
# # Testing
#
# -   `python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_pix2pix`
#
# Change the `--dataroot`, `--name`, and `--direction` to be consistent with your trained model's configuration and how you want to transform images.
#
# > from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
# > Note that we specified --direction BtoA as Facades dataset's A to B direction is photos to labels.
#
# > If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use --model test option. See ./scripts/test_single.sh for how to apply a model to Facade label maps (stored in the directory facades/testB).
#
# > See a list of currently available models at ./scripts/download_pix2pix_model.sh

# + colab={} colab_type="code" id="mey7o6j-0368"
# !ls checkpoints/

# + colab={} colab_type="code" id="uCsKkEq0yGh0"
# !python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_label2photo_pretrained --use_wandb

# + [markdown] colab_type="text" id="OzSKIPUByfiN"
# # Visualize

# + colab={} colab_type="code" id="9Mgg8raPyizq"
import matplotlib.pyplot as plt

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_fake_B.png')
plt.imshow(img)

# + colab={} colab_type="code" id="0G3oVH9DyqLQ"
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_A.png')
plt.imshow(img)

# + colab={} colab_type="code" id="ErK5OC1j1LH4"
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_B.png')
plt.imshow(img)
