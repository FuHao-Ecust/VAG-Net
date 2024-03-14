import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import kornia as K
BN_EPS = 1e-4

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False,
                 is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        if is_bn:
            if out_channels // num_groups == 0:
                num_groups = 1
            self.gn = nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        if is_relu is False: self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, num_groups=num_groups),
        )
        self.pooling = pooling
    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv, pool
        else:
            return conv

class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(M_Conv, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        conv = self.encode(x)
        return conv


class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, bn=False,num_groups=32):
        super(M_Decoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn,  num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, num_groups=num_groups),
        )

    def forward(self, x):
        x = self.decode(x)
        return x

class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # Mean
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A * hr_x + mean_b).float()

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r
    def forward(self, x):
        assert x.dim() == 4
        x = K.filters.gaussian_blur2d(x, (11, 11), (11.0, 11.0))
        return x

class GridAttentionModule(nn.Module):
    def __init__(self,in_channels):
        super(GridAttentionModule, self).__init__()
        self.theta = AttentionModule(in_channels)
        self.psi = nn.Conv2d(in_channels,out_channels=1,kernel_size=1,stride=1)
    def forward(self,x,g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        g = self.theta(g)
        phi_g = F.interpolate(g,size=theta_x_size[2:],mode='bilinear')
        f = F.gelu(theta_x + phi_g)
        f = torch.sigmoid(self.psi(f))
        return f

class AttentionModule(nn.Module): #LKA
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class VAG_Net(nn.Module):
    def __init__(self, n_classes, input_channel=1, bn=True):
        super(VAG_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(input_channel, 64, kernel_size=3)
        self.conv3 = M_Conv(input_channel, 128, kernel_size=3)
        self.conv4 = M_Conv(input_channel, 256, kernel_size=3)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(input_channel, 32, kernel_size=3, bn=bn)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512, 256, kernel_size=3, bn=bn)
        self.up6 = M_Decoder(256, 128, kernel_size=3, bn=bn)
        self.up7 = M_Decoder(128, 64, kernel_size=3, bn=bn)
        self.up8 = M_Decoder(64, 32, kernel_size=3, bn=bn)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the guided filter
        self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)

        # attention blocks
        self.attentionblock5 = GridAttentionModule(in_channels=512)
        self.attentionblock6 = GridAttentionModule(in_channels=256)
        self.attentionblock7 = GridAttentionModule(in_channels=128)
        self.attentionblock8 = GridAttentionModule(in_channels=64)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.interpolate(x, size=(img_shape // 2, img_shape // 2), mode='bilinear')
        x_3 = F.interpolate(x, size=(img_shape // 4, img_shape // 4), mode='bilinear')
        x_4 = F.interpolate(x, size=(img_shape // 8, img_shape // 8), mode='bilinear')

        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        FG = torch.cat([self.conv4(x_4), conv4], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.interpolate(FG, size=(H // 2, W // 2), mode='bilinear')
        out = self.gf(FG_small, out, FG, self.attentionblock5(FG_small, out))
        up5 = self.up5(out) # [32,32]

        FG = torch.cat([self.conv3(x_3), conv3], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.interpolate(FG, size=(H // 2, W // 2), mode='bilinear')
        out = self.gf(FG_small, up5, FG, self.attentionblock6(FG_small, up5))
        up6 = self.up6(out) #[64,64]

        FG = torch.cat([self.conv2(x_2), conv2], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.interpolate(FG, size=(H // 2, W // 2), mode='bilinear')
        out = self.gf(FG_small, up6, FG, self.attentionblock7(FG_small, up6))
        up7 = self.up7(out) #[128,128]

        FG = torch.cat([conv1, conv1], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.interpolate(FG, size=(H // 2, W // 2), mode='bilinear')
        out = self.gf(FG_small, up7, FG, self.attentionblock8(FG_small, up7))
        up8 = self.up8(out) #[256,256]

        side_5 = F.interpolate(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.interpolate(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.interpolate(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.interpolate(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return ave_out, side_5, side_6, side_7, side_8

