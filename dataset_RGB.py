import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir,'mask')))


        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]


        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        mask_img = Image.open(mask_path)


        mask_img_2 = mask_img.resize((int(128), int(128)))
        mask_img_3 = mask_img.resize((int(64),int(64)))


        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
            mask_img = TF.pad(mask_img,(0,0,padw,padh), padding_mode='reflect' )

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        mask_img = TF.to_tensor(mask_img)
        mask_img_2 = TF.to_tensor(mask_img_2)
        mask_img_3 = TF.to_tensor(mask_img_3)

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            mask_img = mask_img.flip(1)
            mask_img_2 = mask_img_2.flip(1)
            mask_img_3 = mask_img_3.flip(1)

        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            mask_img = mask_img.flip(2)
            mask_img_2 = mask_img_2.flip(2)
            mask_img_3 = mask_img_3.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
            mask_img = torch.rot90(mask_img,dims=(1,2))
            mask_img_2 = torch.rot90(mask_img_2,dims=(1, 2))
            mask_img_3 = torch.rot90(mask_img_3, dims=(1, 2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            mask_img = torch.rot90(mask_img,dims=(1,2), k=2)
            mask_img_2 = torch.rot90(mask_img_2, dims=(1, 2), k=2)
            mask_img_3 = torch.rot90(mask_img_3, dims=(1, 2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=3)
            mask_img_2 = torch.rot90(mask_img_2, dims=(1, 2), k=3)
            mask_img_3 = torch.rot90(mask_img_3, dims=(1, 2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            mask_img = torch.rot90(mask_img.flip(1),dims=(1,2))
            mask_img_2 = torch.rot90(mask_img_2.flip(1), dims=(1, 2))
            mask_img_3 = torch.rot90(mask_img_3.flip(1), dims=(1, 2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
            mask_img = torch.rot90(mask_img.flip(2), dims=(1, 2))
            mask_img_2 = torch.rot90(mask_img_2.flip(2), dims=(1, 2))
            mask_img_3 = torch.rot90(mask_img_3.flip(2), dims=(1, 2))
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, mask_img, mask_img_2,mask_img_3,filename


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        mask_img = Image.open(mask_path)

        image_transform_2 = transforms.Compose([
            transforms.Resize(128),
        ])
        image_transform_3 = transforms.Compose([
            transforms.Resize(64),
        ])
        mask_img_2 = image_transform_2(mask_img)
        mask_img_3 = image_transform_3(mask_img)


        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        mask_img = TF.to_tensor(mask_img)
        mask_img_2 = TF.to_tensor(mask_img_2)
        mask_img_3 = TF.to_tensor((mask_img_3))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, mask_img, mask_img_2,mask_img_3,filename


