import torch
import numpy as np
import cv2
from skimage.measure import compare_ssim as ski_ssim


def torchSSIM(tar_img, pre_img):

    output = output.data.cpu().numpy()[0]
    ssim += ski_ssim(pre_img, tar_img, data_range=1, multichannel=True)
    return test_ssim

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
  #  cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(filepath, img)
def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps
