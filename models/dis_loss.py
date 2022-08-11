import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from util.image_pool import ImagePool


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.to(self.device)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss(nn.Module):

    def name(self):
        return 'GAN / DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, pred_fake):
        # First, G(A) should fake the discriminator
        return self.criterionGAN(pred_fake, True)

    def get_loss(self, pred_fake, pred_real):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        # pred_fake = self.netD((self.fake.detach() + 0.5))
        loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def __call__(self, pred_fake, pred_real):
        return self.get_loss(pred_fake, pred_real)


class DiscLossLS(nn.Module):
    def name(self):
        return 'lsgan / DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, pred_fake):
        # First, G(A) should fake the discriminator
        return self.criterionGAN(pred_fake, True)

    def get_loss(self, pred_fake, pred_real):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def __call__(self, pred_fake, pred_real):
        return self.get_loss(pred_fake, pred_real)


class RelativisticDiscLoss(nn.Module):
    def name(self):
        return 'ragan / RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()
        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, pred_fake, pred_real):
        # First, G(A) should fake the discriminator

        # Real
        errG = (self.criterionGAN(pred_real - torch.mean(self.fake_pool.query()), 0) +
                self.criterionGAN(pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, pred_fake, pred_real):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_pool.add(pred_fake)
        # Real
        self.real_pool.add(pred_real)
        # Combined loss
        loss_D = (self.criterionGAN(pred_real - torch.mean(self.fake_pool.query()), 1) +
                  self.criterionGAN(pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return loss_D

    def __call__(self, pred_fake, pred_real):
        return self.get_loss(pred_fake, pred_real)


class RelativisticDiscLossLS(nn.Module):
    def name(self):
        return 'ragan-ls / RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images\
        self.real_pool = ImagePool(50)

    def get_g_loss(self, pred_fake, pred_real):
        # First, G(A) should fake the discriminator
        # Real
        errG = (torch.mean((pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
                torch.mean((pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, pred_fake, pred_real):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_pool.add(pred_fake)
        # Real
        self.real_pool.add(pred_real)

        # Combined loss
        loss_D = (torch.mean((pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
                  torch.mean((pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return loss_D

    def __call__(self, pred_fake, pred_real):
        return self.get_loss(pred_fake, pred_real)


class DiscLossWGANGP(nn.Module):
    def name(self):
        return 'wgan-gp / DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        D_fake = net.forward(fakeB)
        return -D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        D_fake = net.forward(fakeB.detach())
        D_fake = D_fake.mean()

        # Real
        D_real = net.forward(realB)
        D_real = D_real.mean()
        # Combined loss
        loss_D = D_fake - D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return loss_D + gradient_penalty

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class DiscLossWGANGPLabel(nn.Module):

    def name(self):
        return 'wgan-gp / DiscLossWGAN-GP'

    def __init__(self, device):
        super(DiscLossWGANGPLabel, self).__init__()
        self.LAMBDA = 10
        self.device = device

    def get_g_loss(self, pred_fake, fake_label):
        # First, G(A) should fake the discriminator
        return -(pred_fake.mean(dim=(1, 2, 3)) * fake_label).mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        if self.device:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.device:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)
        output = torch.ones(disc_interpolates.size())
        if self.device:
            output = output.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  # # grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  # grad_outputs=torch.ones(disc_interpolates.size()),
                                  grad_outputs=output,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fake, real, fake_label, real_label):
        D_fake = net.forward(fake.detach())
        D_fake = (D_fake.mean(dim=(1, 2, 3)) * fake_label).mean()

        # Real
        D_real = net.forward(real)
        D_real = (D_real.mean(dim=(1, 2, 3)) * real_label).mean()
        # Combined loss
        loss_D = D_fake - D_real

        gradient_penalty = self.calc_gradient_penalty(net, real.data, fake.data)
        return loss_D + gradient_penalty

    def __call__(self, net, fake, real, fake_label, real_label):
        return self.get_loss(net, fake, real, fake_label, real_label)


def get_dis_loss(dis_loss_mode):
    if dis_loss_mode == 'gan':
        disc_loss = DiscLoss()
    elif dis_loss_mode == 'lsgan':
        disc_loss = DiscLossLS()
    elif dis_loss_mode == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif dis_loss_mode == 'ragan':
        disc_loss = RelativisticDiscLoss()
    elif dis_loss_mode == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS()
    else:
        raise ValueError("GAN Loss [%s] not recognized." % dis_loss_mode)
    return disc_loss
