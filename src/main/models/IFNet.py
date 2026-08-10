
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ac = nn.GELU()

    def forward(self, x):
        out = sum(x)
        out = self.ac(out)
        return out

class InterFre_summary(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ac = nn.GELU()

    def forward(self, x1, x2):
        out = x1+x2
        out = self.ac(out)
        return out


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class IFNet(nn.Module):
    def __init__(self,
                 num_channels: int = 22,
                 out_channels: int = 64,
                 num_classes: int = 3,
                 num_samples: int = 301,
                 radix: int = 2,
                 kernel_size: int = 63,
                 patch_size: int = 125,
                 Lineat_input: int = 256
                 ):
        nn.Module.__init__(self)

        self.num_channels = num_channels
        self.num_class = num_classes
        self.num_samples = num_samples
        self.out_channels = out_channels
        self.radix = radix
        self.mid_channels = self.out_channels * self.radix
        self.kernel_size = kernel_size
        self.patch_size = patch_size

        self.spatial_conv1 = Conv(
            nn.Conv1d(self.num_channels, self.mid_channels, kernel_size=1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_channels),
            activation=None)

        self.tmp_conv = nn.ModuleList()
        for _ in range(self.radix):
            self.tmp_conv.append(Conv(
                nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, 1, groups=self.out_channels,
                          padding=self.kernel_size // 2,  bias=False),
                bn=nn.BatchNorm1d(self.out_channels),
                activation=None))
            self.kernel_size //= 2

        self.inter_fre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.25)

        self.fc = nn.Sequential(
            LinearWithConstraint(Lineat_input, num_classes, doWeightNorm=True),
        )


    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        display = False
        N, C, T = x.shape
        if display: print("input", x.shape)
        out = self.spatial_conv1(x)
        if display: print("sconv1", out.shape)
        out = torch.split(out, self.out_channels, dim=1)

        out = [m(x) for x, m in zip(out, self.tmp_conv)]
        if display: print("tconv0", out[0].shape)
        if display: print("tconv1", out[1].shape)
        out = self.inter_fre(out)
        if display: print("interFre", out.shape)
        out = out.reshape(N, self.out_channels, T//self.patch_size, self.patch_size)
        if display: print("reshape", out.shape)
        out = self.power(out)
        if display: print("power", out.shape)
        out = self.dp(out)
        out = out.flatten(1)
        if display: print("flatten", out.shape)
        out = self.fc(out)
        return out


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ac = nn.GELU()

    def forward(self, x):
        out = sum(x)
        out = self.ac(out)
        return out


class IFNet_summary(nn.Module):
    def __init__(self,
                 num_channels: int = 22,
                 out_channels: int = 64,
                 num_classes: int = 3,
                 num_samples: int = 301,
                 radix: int = 2,
                 kernel_size: int = 63,
                 patch_size: int = 125,
                 Lineat_input: int = 256
                 ):
        nn.Module.__init__(self)

        self.num_channels = num_channels
        self.num_class = num_classes
        self.num_samples = num_samples
        self.out_channels = out_channels
        self.radix = radix
        self.mid_channels = self.out_channels * self.radix
        self.kernel_size = kernel_size
        self.patch_size = patch_size

        self.spatial_conv1 = Conv(
            nn.Conv1d(self.num_channels, self.mid_channels, kernel_size=1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_channels),
            activation=None)

        self.tmp_conv = nn.ModuleList()
        self.tmp_conv_1 = Conv(
                nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, 1, groups=self.out_channels,
                          padding=self.kernel_size // 2,  bias=False),
                bn=nn.BatchNorm1d(self.out_channels),
                activation=None)
        self.tmp_conv_2 = Conv(
                nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size // 2, 1, groups=self.out_channels,
                          padding=self.kernel_size // 2 // 2,  bias=False),
                bn=nn.BatchNorm1d(self.out_channels),
                activation=None)
        for _ in range(self.radix):
            self.tmp_conv.append(Conv(
                nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, 1, groups=self.out_channels,
                          padding=self.kernel_size // 2,  bias=False),
                bn=nn.BatchNorm1d(self.out_channels),
                activation=None))
            self.kernel_size //= 2

        self.inter_fre = InterFre_summary()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.25)

        self.fc = nn.Sequential(
            LinearWithConstraint(Lineat_input, num_classes, doWeightNorm=True),
        )


    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        display = True
        N, C, T = x.shape
        out = self.spatial_conv1(x)
        out = torch.split(out, self.out_channels, dim=1)
        out1 = self.tmp_conv_1(out[0])
        out2 = self.tmp_conv_2(out[1])

        out = self.inter_fre(out1, out2)
        if display: print("interFre", out.shape)
        out = out.reshape(N, self.out_channels, T//self.patch_size, self.patch_size)
        if display: print("reshape", out.shape)
        out = self.power(out)
        if display: print("power", out.shape)
        out = self.dp(out)
        out = out.flatten(1)
        if display: print("flatten", out.shape)
        out = self.fc(out)
        return out


def test():
    pass


if __name__ == '__main__':
    from torchsummary import summary
    m = IFNet_summary(num_channels= 124,
                 num_classes= 2,
                 num_samples= 750,
                 Lineat_input= 384)
    summary(m, input_size=(124, 750), batch_size=1, device='cpu')

    from fvcore.nn import FlopCountAnalysis
    inputs = torch.randn(1, 124, 750)
    flop_counter = FlopCountAnalysis(m, inputs)
    print(f"FLOPs: {flop_counter.total()}")

"""
Version history:

"""
