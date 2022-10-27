import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
#        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = ''
        self.modlename = 'training1'

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

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
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


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