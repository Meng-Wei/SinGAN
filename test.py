from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import SinGAN.models as models
import os
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data
import math
import matplotlib.pyplot as plt
import numpy as np

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
parser.add_argument('--input_name', help='input image name', default='big.jpg')
#==========================
parser.add_argument('--pyramid', type=bool, 
                    help='training the model to fit the Laplacian Pyramid', default=False)
#==========================
# Modes including:
# train
# random_samples
# random_samples_arbitrary_sizes
# harmonization
# editing
# SR_train: super resolution train
# SR: super resolution
# paint_train
# paint2image
# animation_train
# animation
parser.add_argument('--mode', help='task to be done', default='train')
opt = parser.parse_args()
opt = functions.post_config(opt)
opt.scale_factor = 0.5
opt.max_size = 1500

real = functions.read_image(opt)
functions.adjust_scales2image(real, opt)
reals = []
reals = functions.creat_reals_pyramid(real,reals,opt)
next_img = np.zeros((1))

for i in range(len(reals) - 1):
    cur_img = reals[i]
    next_img = reals[i+1]
    a, b, c, d = next_img.shape
    fake_img = imresize_to_shape(cur_img,(c, d, b),opt)
    diff = (next_img - fake_img).abs() - 1
    # print(fake_img.shape, torch.max(fake_img), torch.min(fake_img))
    # print(next_img.shape, torch.max(next_img), torch.min(next_img), '\n')
    plt.imsave('outputs/upsample%i.png' % i, functions.convert_image_np(fake_img.detach()), vmin=0, vmax=1)
    plt.imsave('outputs/diff%i.png' % i, functions.convert_image_np(diff.detach()), vmin=0, vmax=1)