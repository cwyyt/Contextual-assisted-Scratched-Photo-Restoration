
import torch
import torch.nn as nn
from Discriminator import NLayerDiscriminator
import functools
from torch.nn import init

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x

        return res
############################################################################
class shallow_feat(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction,bias, act):
        super(shallow_feat, self).__init__()
        self.conv_shallow = conv(3, n_feat, kernel_size, bias=bias)
        self.CAB_shallow = CAB(n_feat, kernel_size, reduction, bias, act)
    def forward(self,x):
        x=self.conv_shallow(x)
        x=self.CAB_shallow(x)

        return x
##########################################################################
class MGA_32(nn.Module):
    def \
            __init__(self, n_feat, kernel_size, bias):
        super(MGA_32, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.up_2 = UpSample_no_conv_2(80, 60)
        self.down_2 = DownSample_2(40,60)
    def forward(self, enc3, enc2, mask):

        x1 = enc3*mask[2]
        x1_up = self.up_2(x1)
        x2 = enc2*mask[1]
        x2_c = self.conv1(x2)
        img = torch.abs(x1_up - x2_c)
        x3 = 2 / (1 + torch.exp(-img))
        x2 = x2*x3
        x4 = x2+enc2*(1-mask[1])

        return x4
class MGA_21(nn.Module):
    def \
            __init__(self, n_feat, kernel_size, bias):
        super(MGA_21, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.up_4 = UpSample_no_conv_4_3(80, 40)
        self.down_4 = DownSample_4(40,80)

    def forward(self, enc3, enc1, mask):
        x1 = enc3 * mask[2]
        x1_up = self.up_4(x1)
        x2 = enc1*mask[0]
        x2 = self.conv1(x2)
        img = torch.abs(x1_up-x2)
        x3 = 2 / (1 + torch.exp(-img))
        x2 = x2*x3
        x4 = x2+enc1*(1-mask[0])

        return x4


################################################################################
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):

        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()
        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.MGA_32 = MGA_32(60, kernel_size, bias=bias)
        self.MGA_21 = MGA_21(40,  kernel_size, bias=bias)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats, kernel_size, bias=bias)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats, kernel_size, bias=bias)


    def forward(self, outs, mask):
        enc1, enc2, enc3  = outs

        dec3 = self.decoder_level3(enc3)
        x_32 = self.MGA_32(enc3, enc2, mask)
        x_32 = self.up32(dec3, x_32)

        dec2 = self.decoder_level2(x_32)
        x_21 = self.MGA_21(enc3, enc1, mask)
        x_21 = self.up21(dec2, x_21)

        dec1 = self.decoder_level1(x_21)
        return [dec1, dec2, dec3]



class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):

        input = inputt[0]
        mask = inputt[1]
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.byte(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.byte(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.byte(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)

        return out

class PCconv(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, scale_unetfeats1=2,scale_unetfeats2=4,stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PCconv, self).__init__()
        self.conv3 = conv(80, out_channels, kernel_size, bias=bias)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(80, 80, innorm=True)]
            seuqence_5 += [PCBActiv(80, 80, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(80, 80, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.conv6 = conv(240, 80, kernel_size, bias=bias)


    def forward(self, input, mask):


        mask[2] = torch.add(torch.neg(mask[2].float()), 1)
        x_1 = [input[2]*mask[2], mask[2]]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_1)
        x_DE_5 = self.cov_5(x_1)
        x_DE_7 = self.cov_7(x_1)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        x_1 = self.conv6(x_DE_fuse)+input[2]*mask[2]
        out = [input[0], input[1], x_1]
        return out

class PCblock(nn.Module):
    def __init__(self, stde_list, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PCblock, self).__init__()
        self.pc_block = PCconv(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, input, mask):
        out = self.pc_block(input, mask)
        return out

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class DownSample_2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DownSample_2, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class DownSample_4(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DownSample_4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample_no_conv_4(nn.Module):
    def __init__(self, in_channels=3,out_channels=40):
        super(UpSample_no_conv_4, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample_no_conv_2(nn.Module):
    def __init__(self, in_channels=3,out_channels=60):
        super(UpSample_no_conv_2, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels,out_channels, 1,stride=1, padding=0, bias=False ))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample_no_conv_4_3(nn.Module):
        def __init__(self, in_channels=3, out_channels=40):
            super(UpSample_no_conv_4_3, self).__init__()
            self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

        def forward(self, x):
            x = self.up(x)
            return x
class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor, kernel_size, bias):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))
        self.conv = conv(in_channels*2, in_channels,kernel_size, bias=bias)
    def forward(self, x,y):
        x = self.up(x)
    #    x_cat = torch.cat([x,y], 1)  #concat
    #    x_cat = self.conv(x_cat)
        x = x + y
        return x


########################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, gain=init_gain)
    return net

def define_D(input_nc=3, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=False)
    return init_net(netD, init_type, init_gain, gpu_ids)

def define_G(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4,bias=False, norm='batch',init_type='normal',init_gain=0.02,gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)
    Scratch = ScratchNet(in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab, kernel_size, reduction,bias)
    return init_net(Scratch, init_type, init_gain, gpu_ids)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
##########################################################################

class ScratchNet(nn.Module):
    def __init__(self, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, kernel_size=3, reduction=4,bias=False):
        super(ScratchNet, self).__init__()
        act=nn.PReLU()
        self.shallow_feat1 = shallow_feat(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        stde_list = []
        self.PCblock = PCconv(stde_list, 3, 3, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.conv1 = conv(40, 3, kernel_size, bias=bias)
        self.conv2 = conv(60, 3, kernel_size, bias=bias)
        self.conv1_1 = conv(40, 1, kernel_size, bias=bias)
        self.conv2_1 = conv(60, 1, kernel_size, bias=bias)
        self.conv3 = conv(80, 3, kernel_size, bias=bias)



        self.up_2 = UpSample_no_conv_2(60, 40)
        self.up_4 = UpSample_no_conv_4_3(80, 40)
        self.up_4_1 = UpSample_no_conv_4_3(80, 3)
        self.down_2 = DownSample_2(40,60)
        self.down_4 = DownSample_4(40,80)
        self.conv_MGCA = conv(80, 40, kernel_size, bias=bias)

        self.concat_MGCA = conv(120, n_feat, kernel_size, bias=bias)
        self.concat  = conv(80, 40, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)

        self.tail     = conv(n_feat+scale_orsnetfeats, 3, kernel_size, bias=bias)

        self.up_2_layer_loss = UpSample_no_conv_2(60, 3)
        self.up_4_layer_loss = UpSample_no_conv_4_3(80, 3)


    def forward(self, x3_img, x3_mask,x3_mask_2,x3_mask_3):

        x1 = self.shallow_feat1(x3_img)
        feat1 = self.stage1_encoder(x1)

        mask_list = []
        mask_1 = x3_mask.squeeze(0).resize_(256,256)
        mask_1 = mask_1.repeat(1,40,1,1)
        mask_list.append(mask_1)

        mask_2 = x3_mask_2.squeeze(0).resize_(128,128)
        mask_2 = mask_2.repeat(1,60,1,1)
        mask_list.append(mask_2)

        mask_3 = x3_mask_3.squeeze(0).resize_(64,64)
        mask_3 = mask_3.repeat(1,80,1,1)
        mask_list.append(mask_3)

        inpaint_out = self.PCblock(feat1, mask_list)
        res1 = self.stage1_decoder(inpaint_out, mask_list)
        stage1_img = self.conv1(res1[0])
        stage1_img = stage1_img+x3_img

        res1_1= self.up_2_layer_loss(res1[1])
        res2_2= self.up_4_layer_loss(res1[2])
        return [stage1_img,res1_1,res2_2]

