
from torch.optim import lr_scheduler
import torch
torch.backends.cudnn.benchmark = True
from ScratchNet import define_D
from ScratchNet import ScratchNet
import losses
import torch
import random
from losses import VGG16, PerceptualLoss, StyleLoss, GANLoss
import torch.nn as nn
from base_model import get_scheduler
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import os
from collections import OrderedDict

class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""

    def __init__(self, channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel, channel * 2, 3, padding=1, bias=False)
        self.rbn1 = nn.BatchNorm2d(channel * 2)
        self.rconv2 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self, feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature))))))



class Restoration(nn.Module):
    def __init__(self, opt):
        super(Restoration, self).__init__()
        self.isTrain = True
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.device = torch.device('cuda')
        self.save_dir = '/media/b3-542/disk/some_code/inpainting/scratch_contexted_old_photo_restoration/crack_inpainting/crack_inpainting/checkpoints/crack_inpainting/models/MPRNet+Dis'
        self.modlename = 'training1'
        # define tensors
        self.vgg = VGG16()
        self.mprnet = ScratchNet().cuda()
        self.PerceptualLoss = PerceptualLoss()
        self.StyleLoss = StyleLoss()
        self.input = self.Tensor()
        self.target = self.Tensor()
        self.Gt_Local = self.Tensor()
        self.mask_global = self.Tensor()
        self.model_names = []
        self.netG = self.mprnet
        self.model_names = ['G']
        if self.isTrain:
                self.netD = define_D(input_nc=3, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[0], init_gain=0.02)
                self.netF = define_D(input_nc=3, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[0], init_gain=0.02)
                self.model_names.append('D')
                self.model_names.append('F')
        if self.isTrain:
            self.old_lr = 0.0002
            # define loss functions
            self.criterionGAN = losses.GANLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.PerceptualLoss = losses.PerceptualLoss()
            self.StyleLoss = losses.StyleLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=0.0002, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))



    def name(self):
        return self.modlename

    def set_input(self, input, target, mask, mask_2, mask_3):

        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.mask = mask.to(self.device)
        self.mask_2 = mask_2.to(self.device)
        self.mask_3 = mask_3.to(self.device)
        self.Gt_Local = target.to(self.device)
        # define local area which send to the local discriminator
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.Gt_Local = self.Gt_Local[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]

    def forward(self):

        self.fake_out = self.netG(self.input, self.mask, self.mask_2, self.mask_3)

    def backward_D(self):

        fake_AB = self.fake_out[0]
        real_AB = self.target
        real_local = self.Gt_Local
        fake_local = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]

        # Global Discriminator
        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)
        loss_D_fake =self.criterionGAN(pred_fake, pred_real, True)

        # Local discriminator
        pred_fake_F = self.netF(fake_local.detach())
        pred_real_F = self.netF(real_local)
        loss_F_fake = self.criterionGAN(pred_fake_F, pred_real_F, True)

        loss_D = loss_D_fake+loss_F_fake
        loss_D.backward()

    def backward_G(self):
        # First, The generator should fake the discriminator
        real_AB = self.target
        fake_AB = self.fake_out[0]
        real_local = self.Gt_Local
        fake_local = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)
        # Local discriminator
        pred_real_F = self.netF(real_local)
        pred_fake_f = self.netF(fake_local)

        real_AB_2 = self.target
        fake_AB_2 = self.fake_out[1]
        real_local_2 = self.Gt_Local
        fake_local_2 = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real_2 = self.netD(real_AB_2)
        pred_fake_2 = self.netD(fake_AB_2)
        # Local discriminator
        pred_real_F_2 = self.netF(real_local_2)
        pred_fake_f_2 = self.netF(fake_local_2)


        real_AB_3 = self.target
        fake_AB_3 = self.fake_out[2]
        real_local_3 = self.Gt_Local
        fake_local_3 = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real_3 = self.netD(real_AB_3)
        pred_fake_3 = self.netD(fake_AB_3)
        # Local discriminator
        pred_real_F_3 = self.netF(real_local_3)
        pred_fake_f_3 = self.netF(fake_local_3)


        self.loss_G_GAN_1 = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F, False)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_2, pred_real_2, False) + self.criterionGAN(pred_fake_f_2, pred_real_F_2, False)
        self.loss_G_GAN_3 = self.criterionGAN(pred_fake_3, pred_real_3, False) + self.criterionGAN(pred_fake_f_3,
                                                                                                   pred_real_F_3, False)

        # Second, Reconstruction loss
        self.loss_L1_1 = self.criterionL1(self.fake_out[0], self.target)
        self.loss_L1_2 = self.criterionL1(self.fake_out[1], self.target)
        self.loss_L1_3 = self.criterionL1(self.fake_out[2], self.target)


        self.Perceptual_loss_1 = self.PerceptualLoss(self.fake_out[0], self.target)
        self.Perceptual_loss_2 = self.PerceptualLoss(self.fake_out[1], self.target)
        self.Perceptual_loss_3 = self.PerceptualLoss(self.fake_out[2], self.target)


        self.Style_Loss_1 = self.StyleLoss(self.fake_out[0], self.target)
        self.Style_Loss_2 = self.StyleLoss(self.fake_out[1], self.target)
        self.Style_Loss_3 = self.StyleLoss(self.fake_out[2], self.target)


        self.stage1_G_loss = self.loss_L1_1 * 1 + self.loss_G_GAN_1 * 0.2 + self.Perceptual_loss_1 * 0.2 + self.Style_Loss_1 * 250
        self.stage2_G_loss = self.loss_L1_2 * 1 + self.loss_G_GAN_2 * 0.2 + self.Perceptual_loss_2 * 0.2 + self.Style_Loss_2 * 250
        self.stage3_G_loss = self.loss_L1_3 * 1 + self.loss_G_GAN_3 * 0.2 + self.Perceptual_loss_3 * 0.2 + self.Style_Loss_3 * 250

        # compute inpint_loss
        self.layer_loss = self.stage1_G_loss+0.5*self.stage2_G_loss+0.25*self.stage3_G_loss
        self.layer_loss.backward(retain_graph=True)

    def backward_G_unalign(self):

        real_AB = self.target
        fake_AB = self.fake_out[0]
        real_local = self.Gt_Local
        fake_local = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)
        # Local discriminator
        pred_real_F = self.netF(real_local)
        pred_fake_f = self.netF(fake_local)

        real_AB_2 = self.target
        fake_AB_2 = self.fake_out[1]
        real_local_2 = self.Gt_Local
        fake_local_2 = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real_2 = self.netD(real_AB_2)
        pred_fake_2 = self.netD(fake_AB_2)
        # Local discriminator
        pred_real_F_2 = self.netF(real_local_2)
        pred_fake_f_2 = self.netF(fake_local_2)

        real_AB_3 = self.target
        fake_AB_3 = self.fake_out[2]
        real_local_3 = self.Gt_Local
        fake_local_3 = fake_AB[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real_3 = self.netD(real_AB_3)
        pred_fake_3 = self.netD(fake_AB_3)
        # Local discriminator
        pred_real_F_3 = self.netF(real_local_3)
        pred_fake_f_3 = self.netF(fake_local_3)

        self.loss_G_GAN_1 = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
                                                                                               False)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_2, pred_real_2, False) + self.criterionGAN(pred_fake_f_2,
                                                                                                   pred_real_F_2, False)
        self.loss_G_GAN_3 = self.criterionGAN(pred_fake_3, pred_real_3, False) + self.criterionGAN(pred_fake_f_3,
                                                                                                   pred_real_F_3, False)


        # Second, Reconstruction loss
        self.loss_L1_1 = self.criterionL1(self.fake_out[0], self.target)
        self.loss_L1_2 = self.criterionL1(self.fake_out[1], self.target)
        self.loss_L1_3 = self.criterionL1(self.fake_out[2], self.target)



        self.Perceptual_loss_1 = self.PerceptualLoss(self.fake_out[0], self.target)
        self.Perceptual_loss_2 = self.PerceptualLoss(self.fake_out[1], self.target)
        self.Perceptual_loss_3 = self.PerceptualLoss(self.fake_out[2], self.target)


        self.Style_Loss_1 = self.StyleLoss(self.fake_out[0], self.target)
        self.Style_Loss_2 = self.StyleLoss(self.fake_out[1], self.target)
        self.Style_Loss_3 = self.StyleLoss(self.fake_out[2], self.target)


        self.stage1_G_loss = self.loss_L1_1 * 1 + self.loss_G_GAN_1 * 0.2 + self.Perceptual_loss_1 * 0.2 + self.Style_Loss_1 * 250
        self.stage2_G_loss = self.loss_L1_2 * 1 + self.Perceptual_loss_2 * 0.2 + self.Style_Loss_2 * 250
        self.stage3_G_loss = self.loss_L1_3 * 1 + self.Perceptual_loss_3 * 0.2 + self.Style_Loss_3 * 250

        # compute inpint_loss
        self.layer_loss = self.stage1_G_loss + self.stage2_G_loss + self.stage3_G_loss
        self.layer_loss.backward(retain_graph=True)


    def optimize_parameters(self):
        self.forward()
        # Optimize the D and F first
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        # Optimize EN, DE, MEDEF

        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters_unalign(self):
        self.forward()
        # Optimize the D and F first
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        # Optimize EN, DE, MEDEF

        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G_unalign()
        self.optimizer_G.step()

    # helper saving function that can be used by subclasses
    def save_networks(self, which_epoch):
        for name in self.model_names:
            print(name)
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename).replace('\\', '/')
                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.state_dict(), 'optimize': optimize.state_dict()}, save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(self.device))
                optimize.load_state_dict(state_dict['optimize'])
                net.load_state_dict(state_dict['net'])


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_scheduler(optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + 1 - 20) / float(100 + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def get_current_visuals(self):
        input_image = self.input.data.cpu()
        fake_image_2 = self.fake_out[0].data.cpu()
        real_gt = self.target.data.cpu()
        mask = self.mask.data.cpu()
        return input_image, fake_image_2, real_gt, mask

    def get_current_errors(self):
        # show the current loss
        return OrderedDict([('stage1', self.stage1_G_loss.data),
                            ('stage2', self.stage2_G_loss.data),
                            ('stage3', self.stage3_G_loss.data),
                            ('layer_loss', self.layer_loss.data),
                            ])
