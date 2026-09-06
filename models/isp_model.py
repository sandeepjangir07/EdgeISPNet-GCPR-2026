import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR

from .discriminator import UNetDiscriminatorSN
from .edgeispnet import ECB, EdgeISPNet, GSRM, RepConv, SeqConv3x3
from .losses import CharbonnierLoss, GANLoss, MSSSIMLoss, PerceptualLoss


def init_weights(net):
    for module in net.modules():
        if isinstance(module, SeqConv3x3):
            init.kaiming_normal_(module.k0.data, a=0, mode='fan_in')
            module.b0.data.zero_()
            if module.type == 'conv1x1-conv3x3':
                init.kaiming_normal_(module.k1.data, a=0, mode='fan_in')
                module.b1.data.zero_()
            else:
                init.normal_(module.scale.data, mean=0, std=1e-3)
                init.normal_(module.bias.data, mean=0, std=1e-3)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
    for module in net.modules():
        if isinstance(module, GSRM):
            module.init_identity()
    return net


def build_network(opt):
    cfg = opt['network']
    net = EdgeISPNet(
        in_channel=cfg['in_channel'],
        out_channel=cfg['out_channel'],
        inter_channel=cfg['inter_channel'],
        block_nums=cfg['block_nums'],
        blocks_per_group=cfg['blocks_per_group'],
        scale=opt['datasets']['scale'],
        act_type=cfg.get('act_type', 'gelu'),
        with_idt=cfg.get('with_idt', True),
        use_gsrm=cfg.get('use_gsrm', True),
    )
    if opt['phase'] == 'train':
        init_weights(net)
    return net


class ISPModel:
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

        self.net = build_network(opt).to(self.device)
        counts = self.net.count_parameters()
        logger.info(f"EdgeISPNet parameters | train: {counts['train']:,} | "
                    f"inference: {counts['deploy_total']:,} "
                    f"(core {counts['deploy_core']:,} + GSRM {counts['gsrm']:,})")

        self.as_gan = opt['train']['gan']['enabled'] if opt['phase'] == 'train' else False
        self.netD = None

        if opt['phase'] == 'train':
            self._setup_training()
        self._load_checkpoint()

    def _setup_training(self):
        train_cfg = self.opt['train']
        self.weights = train_cfg['loss_weights']

        self.cri_charb = CharbonnierLoss().to(self.device)
        self.cri_msssim = MSSSIMLoss().to(self.device) if self.weights['ms_ssim'] else None
        self.cri_percep = PerceptualLoss().to(self.device) if self.weights['perceptual'] else None

        self.net.train()
        self.optG = torch.optim.Adam(self.net.parameters(), lr=train_cfg['lr'])
        self.schedulers = [MultiStepLR(self.optG, milestones=train_cfg['lr_steps'],
                                       gamma=train_cfg['lr_gamma'])]

        if self.as_gan:
            self.netD = UNetDiscriminatorSN(
                num_in_ch=self.opt['network']['out_channel'],
                num_feat=train_cfg['gan']['num_feat']).to(self.device)
            self.netD.train()
            self.cri_gan = GANLoss(train_cfg['gan']['type']).to(self.device)
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=train_cfg['gan']['lr'])
            self.schedulers.append(MultiStepLR(self.optD, milestones=train_cfg['lr_steps'],
                                               gamma=train_cfg['lr_gamma']))

    def _load_checkpoint(self):
        path = self.opt['resume']['path']
        if not path:
            self.logger.info('No checkpoint given: starting from scratch.')
            return
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state['net'])
        self.begin_step = state.get('step', 0)
        self.begin_epoch = state.get('epoch', 0)
        self.logger.info(f'Loaded generator from {path} (step {self.begin_step}).')

        if self.opt['phase'] != 'train':
            return
        if self.opt['resume'].get('load_optimizer') and 'optG' in state:
            self.optG.load_state_dict(state['optG'])
        if self.netD is not None and state.get('netD') is not None:
            self.netD.load_state_dict(state['netD'])
            if self.opt['resume'].get('load_optimizer') and 'optD' in state:
                self.optD.load_state_dict(state['optD'])

    def feed_data(self, data):
        self.data = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in data.items()}

    def _forward(self):
        return self.net(x=self.data['LQ'], x_global=self.data['Global'],
                        cam_wb=self.data.get('WB'))

    def optimize_parameters(self):
        losses = OrderedDict()
        target = self.data['HQ']

        if self.as_gan:
            for p in self.netD.parameters():
                p.requires_grad = False

        self.optG.zero_grad()
        self.output = torch.clamp(self._forward(), 0.0, 1.0)

        total = self.weights['charbonnier'] * self.cri_charb(self.output, target)
        losses['charb'] = total

        if self.cri_msssim is not None:
            l_ms = self.weights['ms_ssim'] * self.cri_msssim(self.output, target)
            total = total + l_ms
            losses['ms_ssim'] = l_ms

        if self.cri_percep is not None:
            l_p = self.weights['perceptual'] * self.cri_percep(self.output, target)
            total = total + l_p
            losses['percep'] = l_p

        if self.as_gan:
            l_adv = self.weights['adversarial'] * self.cri_gan(self.netD(self.output), True)
            total = total + l_adv
            losses['adv'] = l_adv

        losses['total'] = total
        total.backward()
        self.optG.step()

        if self.as_gan:
            for p in self.netD.parameters():
                p.requires_grad = True
            self.optD.zero_grad()
            l_d_real = self.cri_gan(self.netD(target), True)
            l_d_real.backward()
            l_d_fake = self.cri_gan(self.netD(self.output.detach()), False)
            l_d_fake.backward()
            self.optD.step()
            losses['d_real'] = l_d_real
            losses['d_fake'] = l_d_fake

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def test(self, global_params=None):
        self.net.eval()
        self.output = torch.clamp(
            self.net(x=self.data['LQ'],
                     x_global=None if global_params is not None else self.data['Global'],
                     cam_wb=self.data.get('WB'),
                     global_params=global_params), 0.0, 1.0)
        if self.opt['phase'] == 'train':
            self.net.train()
        return self.output

    @torch.no_grad()
    def compute_global_params(self, global_view, cam_wb):
        """GSRM is evaluated once per full frame and reused across every tile."""
        self.net.eval()
        return self.net.compute_global_params(global_view.to(self.device),
                                              cam_wb.to(self.device))

    def get_visuals(self):
        out = OrderedDict(output=self.output.detach().float().cpu())
        for key in ('HQ', 'LQ'):
            if key in self.data:
                out[key] = self.data[key].detach().float().cpu()
        return out

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_lr(self):
        return self.optG.param_groups[0]['lr']

    def save(self, epoch, step, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state = {'net': self.net.state_dict(), 'epoch': epoch, 'step': step,
                 'optG': self.optG.state_dict()}
        if self.netD is not None:
            state['netD'] = self.netD.state_dict()
            state['optD'] = self.optD.state_dict()
        path = os.path.join(save_dir, f'edgeispnet_step{step}.pth')
        torch.save(state, path)
        self.logger.info(f'Saved checkpoint: {path}')
