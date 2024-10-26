"""
U-Net
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, e_in, x):
        _, _, e_h, e_w = e_in.size()
        d_x = F.interpolate(x, size=[e_h, e_w], mode='bilinear', align_corners=True)
        c_x = self.conv(d_x)
        c_x = self.norm(c_x)
        c_x = self.relu(c_x)
        d_x = torch.cat((e_in, c_x), dim=1)
        return d_x

class Gate_attention(nn.Module):
    def __init__(self, in_c, out_c):
        super(Gate_attention, self).__init__()
        self.K1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.A = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, d_ft, x):
        att_f = F.relu(self.K1(d_ft))
        att_f = F.sigmoid(self.A(att_f))
        x_att = att_f * x

        return x_att, att_f

"""
    U_Net
"""
class AUNet(nn.Module):

    def __init__(self, sigma=0.2, in_ch=3, out_ch=3):
        super(AUNet, self).__init__()

        # filters
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # gate attention
        self.i1_att_d = Gate_attention(64, 64)
        self.i1_att = Gate_attention(64, 64)

        # Maxpool
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # decoder
        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Buddha_head = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        # Gate attention
        att_feature, att_map = self.i1_att_d(e1, e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d_x4 = self.Up4(e4, e5)
        d_x4 = self.Up_conv4(d_x4)

        d_x3 = self.Up3(e3, d_x4)
        d_x3 = self.Up_conv3(d_x3)

        d_x2 = self.Up2(e2, d_x3)
        d_x2 = self.Up_conv2(d_x2)

        d_x1 = self.Up1(e1, d_x2)
        d_x1 = self.Up_conv1(d_x1)

        Buddha_head = self.Buddha_head(d_x1)

        return Buddha_head, att_map



