import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, in_channels, out_channels, depth_multiplier=1.0):
        super().__init__()
        self.type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.type == 'conv1x1-conv3x3':
            self.mid_channels = int(out_channels * depth_multiplier)
            conv0 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, padding=0)
            self.k0, self.b0 = conv0.weight, conv0.bias
            conv1 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=3)
            self.k1, self.b1 = conv1.weight, conv1.bias
            return

        conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.k0, self.b0 = conv0.weight, conv0.bias
        self.scale = nn.Parameter(torch.randn(out_channels, 1, 1, 1) * 1e-3)
        self.bias = nn.Parameter(torch.randn(out_channels) * 1e-3)

        mask = torch.zeros(out_channels, 1, 3, 3)
        if self.type == 'conv1x1-sobelx':
            mask[:, 0, 0, 0], mask[:, 0, 1, 0], mask[:, 0, 2, 0] = 1.0, 2.0, 1.0
            mask[:, 0, 0, 2], mask[:, 0, 1, 2], mask[:, 0, 2, 2] = -1.0, -2.0, -1.0
        elif self.type == 'conv1x1-sobely':
            mask[:, 0, 0, 0], mask[:, 0, 0, 1], mask[:, 0, 0, 2] = 1.0, 2.0, 1.0
            mask[:, 0, 2, 0], mask[:, 0, 2, 1], mask[:, 0, 2, 2] = -1.0, -2.0, -1.0
        elif self.type == 'conv1x1-laplacian':
            mask[:, 0, 0, 1], mask[:, 0, 1, 0] = 1.0, 1.0
            mask[:, 0, 1, 2], mask[:, 0, 2, 1] = 1.0, 1.0
            mask[:, 0, 1, 1] = -4.0
        else:
            raise ValueError(f'unsupported seq type: {seq_type}')
        self.mask = nn.Parameter(mask, requires_grad=False)

    def _pad_with_bias(self, y0):
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0.0)
        b = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b
        y0[:, :, -1:, :] = b
        y0[:, :, :, 0:1] = b
        y0[:, :, :, -1:] = b
        return y0

    def forward(self, x):
        y0 = self._pad_with_bias(F.conv2d(x, self.k0, self.b0, stride=1))
        if self.type == 'conv1x1-conv3x3':
            return F.conv2d(y0, self.k1, self.b1, stride=1)
        return F.conv2d(y0, self.scale * self.mask, self.bias,
                        stride=1, groups=self.out_channels)

    def rep_params(self):
        device = self.k0.device
        if self.type == 'conv1x1-conv3x3':
            rk = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
            rb = torch.ones(1, self.mid_channels, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rb = F.conv2d(rb, self.k1).view(-1) + self.b1
            return rk, rb

        tmp = self.scale * self.mask
        k1 = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            k1[i, i] = tmp[i, 0]
        rk = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
        rb = torch.ones(1, self.out_channels, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        rb = F.conv2d(rb, k1).view(-1) + self.bias
        return rk, rb


def make_activation(act_type, num_channels):
    if act_type == 'gelu':
        return nn.GELU()
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    if act_type == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    if act_type == 'linear':
        return nn.Identity()
    raise ValueError(f'unsupported activation: {act_type}')


class ECB(nn.Module):
    def __init__(self, in_channels, out_channels, depth_multiplier=2.0,
                 act_type='gelu', with_idt=False, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type
        self.with_idt = with_idt and (in_channels == out_channels)
        self.deploy = deploy

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', in_channels, out_channels, depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', in_channels, out_channels)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', in_channels, out_channels)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', in_channels, out_channels)
        self.act = make_activation(act_type, out_channels)

    def rep_params(self):
        rk, rb = self.conv3x3.weight, self.conv3x3.bias
        for branch in (self.conv1x1_3x3, self.conv1x1_sbx, self.conv1x1_sby, self.conv1x1_lpl):
            k, b = branch.rep_params()
            rk, rb = rk + k, rb + b
        if self.with_idt:
            k_idt = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=rk.device)
            for i in range(self.out_channels):
                k_idt[i, i, 1, 1] = 1.0
            rk = rk + k_idt
        return rk, rb

    def forward(self, x):
        if self.deploy:
            rk, rb = self.rep_params()
            y = F.conv2d(x, rk, rb, stride=1, padding=1)
        else:
            y = (self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x)
                 + self.conv1x1_sby(x) + self.conv1x1_lpl(x))
            if self.with_idt:
                y = y + x
        return self.act(y)


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='gelu', deploy=False):
        super().__init__()
        self.act_type = act_type
        self.deploy = deploy
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.act = make_activation(act_type, out_channels)

    def rep_params(self):
        rk = self.conv3x3.weight + F.pad(self.conv1x1.weight, (1, 1, 1, 1))
        rb = self.conv3x3.bias + self.conv1x1.bias
        return rk, rb

    def forward(self, x):
        if self.deploy:
            rk, rb = self.rep_params()
            y = F.conv2d(x, rk, rb, stride=1, padding=1)
        else:
            y = self.conv3x3(x) + self.conv1x1(x)
        return self.act(y)


class ECBResGroup(nn.Module):
    def __init__(self, channels, n_blocks, depth_multiplier=2.0,
                 act_type='gelu', with_idt=True, deploy=False, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.blocks = nn.ModuleList([
            ECB(channels, channels, depth_multiplier, act_type, with_idt, deploy)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out * self.res_scale + x


class GSRM(nn.Module):
    """Global Scene Representation Module.

    Predicts white-balance gains and per-channel feature modulation (gamma, beta)
    from a downsampled view of the full frame plus the camera white balance.
    Initialised as an identity mapping: gamma=1, beta=0, wb_gains=cam_wb.
    """

    def __init__(self, in_channels=4, inter_channels=8, wb_residual_scale=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.wb_residual_scale = wb_residual_scale

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_style = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, inter_channels * 2),
        )
        self.fc_wb = nn.Sequential(
            nn.Linear(32 + in_channels, 32), nn.ReLU(inplace=True),
            nn.Linear(32, in_channels),
        )
        self.init_identity()

    @torch.no_grad()
    def init_identity(self):
        self.fc_style[-1].weight.zero_()
        self.fc_style[-1].bias[:self.inter_channels].fill_(1.0)
        self.fc_style[-1].bias[self.inter_channels:].zero_()
        self.fc_wb[-1].weight.zero_()
        self.fc_wb[-1].bias.zero_()

    def forward(self, x_global, cam_wb=None):
        feat = self.encoder(x_global)
        gamma, beta = self.fc_style(feat).chunk(2, dim=1)

        if cam_wb is None:
            cam_wb = torch.ones(feat.size(0), self.in_channels, device=feat.device)
        wb_residual = self.fc_wb(torch.cat([feat, cam_wb], dim=1))
        wb_gains = torch.clamp(cam_wb * (1.0 + self.wb_residual_scale * wb_residual), min=0.01)

        gamma = gamma.view(-1, self.inter_channels, 1, 1)
        beta = beta.view(-1, self.inter_channels, 1, 1) * 0.1
        wb_gains = wb_gains.view(-1, self.in_channels, 1, 1)
        return gamma, beta, wb_gains


class EdgeISPNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, inter_channel=8, block_nums=4,
                 blocks_per_group=3, scale=2, act_type='gelu', with_idt=True,
                 use_gsrm=True, deploy=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inter_channel = inter_channel
        self.scale = scale
        self.use_gsrm = use_gsrm
        self.deploy = deploy

        self.gsrm = GSRM(in_channel, inter_channel) if use_gsrm else None
        self.head = ECB(in_channel, inter_channel, 2.0, act_type, with_idt, deploy)
        self.body = nn.ModuleList([
            ECBResGroup(inter_channel, blocks_per_group, 2.0, act_type, with_idt, deploy)
            for _ in range(block_nums)
        ])
        self.expansion = ECB(inter_channel, inter_channel * scale ** 2, 2.5,
                             act_type, False, deploy)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.post_shuffle_conv = nn.Conv2d(inter_channel, inter_channel, 3, padding=1)
        self.post_shuffle_act = nn.GELU()
        self.tail1 = RepConv(inter_channel, inter_channel, act_type, deploy)
        self.tail2 = RepConv(inter_channel, out_channel, 'linear', deploy)

    def set_deploy(self, deploy=True):
        self.deploy = deploy
        for module in self.modules():
            if isinstance(module, (ECB, RepConv)):
                module.deploy = deploy
        return self

    @torch.no_grad()
    def compute_global_params(self, x_global, cam_wb=None):
        """Run the GSRM once per full frame; reuse the result for every tile."""
        if self.gsrm is None:
            return None
        return self.gsrm(x_global, cam_wb=cam_wb)

    def forward(self, x, x_global=None, cam_wb=None, global_params=None):
        gamma = beta = wb_gains = None
        if self.gsrm is not None:
            if global_params is not None:
                gamma, beta, wb_gains = global_params
            elif x_global is not None:
                gamma, beta, wb_gains = self.gsrm(x_global, cam_wb=cam_wb)

        if wb_gains is not None:
            x = x * wb_gains

        feat = self.head(x)
        if gamma is not None:
            feat = feat * gamma + beta

        y = feat
        for group in self.body:
            y = group(y)
        feat = y + feat

        out = self.pixel_shuffle(self.expansion(feat))
        out = self.post_shuffle_act(self.post_shuffle_conv(out))
        out = self.tail2(self.tail1(out))
        return torch.sigmoid(out)

    @torch.no_grad()
    def count_parameters(self):
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        core = sum(p.numel() for p in self.post_shuffle_conv.parameters())
        for module in self.modules():
            if isinstance(module, (ECB, RepConv)):
                rk, rb = module.rep_params()
                core += rk.numel() + rb.numel()
                core += sum(p.numel() for p in module.act.parameters())

        gsrm = sum(p.numel() for p in self.gsrm.parameters()) if self.gsrm else 0
        return {'train': train_params, 'deploy_core': core,
                'gsrm': gsrm, 'deploy_total': core + gsrm}
