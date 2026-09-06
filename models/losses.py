import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def _gaussian_window(window_size, channels, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('weights', torch.tensor(weights))

    def _ssim(self, img1, img2, window, channels):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        s2 = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        s12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2
        cs = (2 * s12 + c2) / (s1 + s2 + c2 + 1e-8)
        lum = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1 + 1e-8)
        return (lum * cs).mean(), cs.mean()

    def ms_ssim(self, img1, img2):
        channels = img1.size(1)
        window = _gaussian_window(self.window_size, channels).to(img1)
        mssim, mcs = [], []
        for _ in range(self.weights.numel()):
            ssim_val, cs_val = self._ssim(img1, img2, window, channels)
            mssim.append(ssim_val)
            mcs.append(cs_val)
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)
        weights = self.weights.to(img1)
        mssim = (F.relu(torch.stack(mssim)) + 1e-6) ** weights
        mcs = (F.relu(torch.stack(mcs)) + 1e-6) ** weights
        return mcs[:-1].prod() * mssim[-1]

    def forward(self, pred, target):
        return 1.0 - self.ms_ssim(pred, target)


class PerceptualLoss(nn.Module):
    """VGG19 relu5_4 feature L1 loss on ImageNet-normalised inputs."""

    def __init__(self, layer_index=35):
        super().__init__()
        vgg = torchvision.models.vgg19(weights='DEFAULT').features[:layer_index + 1]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred), self.vgg(target.detach()))


class GANLoss(nn.Module):
    def __init__(self, gan_type='lsgan'):
        super().__init__()
        self.gan_type = gan_type
        self.loss = nn.MSELoss() if gan_type == 'lsgan' else nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        target = torch.full_like(pred, 1.0 if target_is_real else 0.0)
        return self.loss(pred, target)
