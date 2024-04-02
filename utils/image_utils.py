#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torchvision import transforms
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def rgb2loftrgray(img):
    resizer = transforms.Resize([480,640])
    gray=transforms.functional.rgb_to_grayscale(img)
    img11 = resizer(gray)
    img11 = img11[None].cuda()
    return img11