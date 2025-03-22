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
# <a href="https://colab.research.google.com/github/bkkaggle/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] colab_type="text" id="5VIGyIus8Vr7"
# Take a look at the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for more information

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
# -   `bash ./datasets/download_cyclegan_dataset.sh [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos]`
#
# Or use your own dataset by creating the appropriate folders and adding in the images.
#
# -   Create a dataset folder under `/dataset` for your dataset.
# -   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the `testA` folder, images you want to transform from b to a (dog2cat) in the `testB` folder, and do the same for the `trainA` and `trainB` folders.

# + colab={} colab_type="code" id="vrdOettJxaCc"
# !bash ./datasets/download_cyclegan_dataset.sh horse2zebra

# + [markdown] colab_type="text" id="gdUz4116xhpm"
# # Pretrained models
#
# Download one of the official pretrained models with:
#
# -   `bash ./scripts/download_cyclegan_model.sh [apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower]`
#
# Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

# + colab={} colab_type="code" id="B75UqtKhxznS"
# !bash ./scripts/download_cyclegan_model.sh horse2zebra

# + [markdown] colab_type="text" id="yFw1kDQBx3LN"
# # Training
#
# -   `python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan`
#
# Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. I've found that a batch size of 16 fits onto 4 V100s and can finish training an epoch in ~90s.
#
# Once your model has trained, copy over the last checkpoint to a format that the testing model can automatically detect:
#
# Use `cp ./checkpoints/horse2zebra/latest_net_G_A.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class A to class B and `cp ./checkpoints/horse2zebra/latest_net_G_B.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class B to class A.
#

# + colab={} colab_type="code" id="0sp7TCT2x9dB"
# !python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id -1

# + [markdown] colab_type="text" id="9UkcaFZiyASl"
# # Testing
#
# -   `python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout`
#
# Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.
#
# > from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
# > The option --model test is used for generating results of CycleGAN only for one side. This option will automatically set --dataset_mode single, which only loads the images from one set. On the contrary, using --model cycle_gan requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.
#
# > For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

# + colab={} colab_type="code" id="uCsKkEq0yGh0"
# !python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

# + [markdown] colab_type="text" id="OzSKIPUByfiN"
# # Visualize

# + colab={} colab_type="code" id="9Mgg8raPyizq"
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_fake.png')
plt.imshow(img)

# + colab={} colab_type="code" id="0G3oVH9DyqLQ"
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_real.png')
plt.imshow(img)
