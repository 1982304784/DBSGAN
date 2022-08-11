import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from abc import ABC
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


class BaseModel(ABC):
    def set_requires_grad(self, nets, requires_grad=False):
        """
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

    def tensor2im(self, image_tensor, imtype=np.uint8):
        """
        For tensor in range(0, 0.5)
        :param image_tensor:
        :param imtype:
        :return:
        """
        image_tensor = (image_tensor[0, :, :, :] + 0.5) * 255 + 0.5
        image_tensor = torch.clamp(image_tensor, 0, 255)

        image_numpy = image_tensor.cpu().float().detach().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, fake, real):
        fake = self.tensor2im(fake.data)
        real = self.tensor2im(real.data)

        psnr = PSNR(real, fake)
        ssim = SSIM(real, fake, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        return psnr, ssim

    def DarkChannel(self, img, patch_size=31):
        padsize = int((patch_size - 1) / 2)
        pad = nn.ReflectionPad2d(padsize)
        dark = pad(img)
        dark = dark * -1

        dark = nn.functional.max_pool3d(dark, [3, patch_size, patch_size], stride=1)

        return dark * -1

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(1)

    def gradient_1order(self, x, h_x=None, w_x=None):
        x = self.rgb2gray(x)
        if h_x is None and w_x is None:
            h_x = x.size()[2]
            w_x = x.size()[3]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        grad = torch.cat(((torch.pow((r - l), 2)), torch.pow((t - b), 2)), 1)
        In = nn.InstanceNorm2d(num_features=2, eps=0, affine=False, track_running_stats=False)
        return In(grad)

    def img2dg(self, x):
        return torch.cat((1 - self.gradient_1order(x), 1 - self.DarkChannel(x)), 1)

    def save_images(self, images, root_dir, epoch, iteration):
        filename = root_dir + "/epoch" + str(epoch) + '_' + str(iteration) + '.png'
        torchvision.utils.save_image(images, filename)

    def save_deblur(self, images, root_dir, epoch, iteration):
        blur = images['blur_image'] - 0.5
        if self.opt.cuda:
            blur = blur.cuda()
        fake = self.netG(blur)
        self.save_images(fake, root_dir, epoch, iteration)
