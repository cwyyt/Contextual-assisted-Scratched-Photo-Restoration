
import argparse
from torch.utils.data import DataLoader
from config import Config
import torchvision
opt = Config('training.yml')
from data_RGB import  get_test_data
from Restoration import Restoration
import matplotlib
matplotlib.use('Agg')
import os
import torch


parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--checkpoints_dir', default='', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--which_epoch', type=str, default='43', help='which epoch to load? set to latest to use latest cached model')
args = parser.parse_args()

opt = Config('training.yml')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

test_dataset = get_test_data(args.input_dir, {'patch_size': opt.TRAINING.VAL_PS})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,pin_memory=True)
model = Restoration(opt)
model.load_networks(args.which_epoch)
psnr_val_rgb = []


for ii, data_val in enumerate((test_loader), 0):
        target = data_val[0]
        input_ = data_val[1]
        mask = data_val[2]
        mask_2 = data_val[3]
        mask_3 = data_val[4]
        model.set_input(input_, target, mask, mask_2, mask_3)
        model.forward()
        with torch.no_grad():
            input, output, GT, mask = model.get_current_visuals()
        #    image_out = torch.cat([input, output, GT, mask], 0)
            image_out = output
            grid = torchvision.utils.make_grid(image_out)
            file_name = '/media/b3-542/disk/Datasets/Real_data/test/test_19/results/' + format(str(ii + 1), '0>4s') + '.png'
            torchvision.utils.save_image(grid, file_name, nrow=1)

