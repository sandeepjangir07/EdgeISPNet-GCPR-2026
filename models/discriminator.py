import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=32, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        sn = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        lrelu = lambda t: F.leaky_relu(t, negative_slope=0.2, inplace=True)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)

        x0 = lrelu(self.conv0(x))
        x1 = lrelu(self.conv1(x0))
        x2 = lrelu(self.conv2(x1))
        x3 = lrelu(self.conv3(x2))

        x4 = lrelu(self.conv4(up(x3)))
        if self.skip_connection:
            x4 = x4 + x2
        x5 = lrelu(self.conv5(up(x4)))
        if self.skip_connection:
            x5 = x5 + x1
        x6 = lrelu(self.conv6(up(x5)))
        if self.skip_connection:
            x6 = x6 + x0

        out = lrelu(self.conv7(x6))
        out = lrelu(self.conv8(out))
        return self.conv9(out)
