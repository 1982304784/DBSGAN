import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.base_model import BaseModel
from util.datasets import GoProDataset
from util.metric_counter import MetricCounter
from models.dis_loss import get_dis_loss
from models.discriminator import get_Discriminator
from models.gen_loss import get_gen_loss
from models.generator import get_Generator

parser = argparse.ArgumentParser(description="DBSGAN")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--start_epoch", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--imagesize", type=int, default=256)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-min_lr", type=float, default=0.000001)
parser.add_argument('--input_nc', type=int, default=64)

parser.add_argument('--lambda_content', type=float, default=0.005)
parser.add_argument('--lambda_pixel', type=float, default=1.0)
parser.add_argument('--lambda_adv', type=float, default=0.001)
parser.add_argument('--cuda', type=bool, default=False, help='True means using GPU')

parser.add_argument('--dis_loss_mode', type=str, default='ragan-ls')
parser.add_argument('--gen_loss_mode', type=str, default='perceptual')

args = parser.parse_args()

# Hyper Parameters
METHOD = "DMPHN"
LEARNING_RATE = args.lr
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

ROOT_DIR = 'D:/Deblur/Dataset/GOPRO'
DATA_FILE_DIR = 'datas/GoPro/'
LOG_DIR = 'log_DBSGAN'
CHECKPOINT_DIR = ''
CHECKPOINT_NAME = 'trainModel.pkl'
MODELS_DIR = 'model.pkl'
STEP_TURN = 2


class DMPHNTrainer(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self)
        self.opt = opt
        self.current_loss = {}
        # define networks (both generator and discriminator)
        self.netG = get_Generator()
        self.netD = get_Discriminator(input_c=3, layers=5, norm=nn.InstanceNorm2d, attention=True)
        self.criterion_G = get_gen_loss(self.opt.gen_loss_mode, self.opt.lambda_pixel, self.opt.lambda_content)
        self.criterion_D = get_dis_loss(self.opt.dis_loss_mode)
        if self.opt.cuda:
            self.netG.cuda()
            self.netD.cuda()

        # initialize optimizers, schedulers
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.opt.lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr)
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, eta_min=self.opt.min_lr, T_max=self.opt.epochs,
                                             last_epoch=self.opt.start_epoch)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, eta_min=self.opt.min_lr, T_max=self.opt.epochs,
                                             last_epoch=self.opt.start_epoch)

    def update_G(self, fake, real):
        # update G

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        # calculate loss & gradients
        pred_fake = self.netD(fake)
        pred_real = self.netD(real)

        loss_G_GAN = self.criterion_D.get_g_loss(pred_fake, pred_real) * self.opt.lambda_adv
        self.current_loss['G_adv'] = loss_G_GAN.item()

        loss_content, loss_pixel = self.criterion_G(fake, real)
        self.current_loss['G_content'] = loss_content.item()
        self.current_loss['G_pixel'] = loss_pixel.item()

        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_content + loss_pixel
        self.current_loss['G_loss'] = self.current_loss['G_adv'] + self.current_loss['G_content'] + self.current_loss[
            'G_pixel']
        loss_G.backward()

        # update G's weights
        self.optimizer_G.step()

        self.current_loss['PSNR'], self.current_loss['SSIM'] = self.get_images_and_metrics(fake, real)

    def update_D(self, fake, real):
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        # calculate loss & gradients
        pred_fake = self.netD(fake.detach())
        pred_real = self.netD(real)
        loss_D = self.criterion_D(pred_fake, pred_real) * self.opt.lambda_adv

        self.current_loss['D_loss'] = loss_D.item()
        loss_D.backward()
        self.optimizer_D.step()

    def update_parameters(self, images, mode=1):
        self.current_loss = {}
        blur = Variable(images['blur_image'] - 0.5)
        real = Variable(images['sharp_image'] - 0.5)
        if self.opt.cuda:
            blur = blur.cuda()
            real = real.cuda()

        # compute fake images: G(A)
        fake = self.netG(blur)
        if mode != 0:
            self.update_D(fake, real)
        else:
            self.update_D(self.img2dg(fake), self.img2dg(real))

        fake = self.netG(blur)
        self.update_G(fake, real)

        return self.current_loss

    def Validation(self, images):
        current_loss = {}
        blur = images['blur_image'] - 0.5
        real = images['sharp_image'] - 0.5
        if self.opt.cuda:
            blur = blur.cuda()
            real = real.cuda()

        fake = self.netG(blur)
        current_loss['PSNR'], current_loss['SSIM'] = self.get_images_and_metrics(fake, real)
        return current_loss


def main():
    models = DMPHNTrainer(args)
    metric_counter = MetricCounter(LOG_DIR)

    # ---------------------- load models ----------------------
    if os.path.exists(MODELS_DIR):
        models.netG.load_state_dict(
            torch.load(MODELS_DIR, map_location=torch.device('cpu')))
        print("load models success")

    models.netG = nn.DataParallel(models.netG)

    # ---------------------- load dataset ----------------------
    train_dataset = GoProDataset(
        blur_image_files=DATA_FILE_DIR + 'train_blur_file.txt',
        sharp_image_files=DATA_FILE_DIR + 'train_sharp_file.txt',
        root_dir=ROOT_DIR,
        crop=True,
        crop_size=IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = GoProDataset(
        blur_image_files=DATA_FILE_DIR + 'test_blur_file.txt',
        sharp_image_files=DATA_FILE_DIR + 'test_sharp_file.txt',
        root_dir=ROOT_DIR,
        crop=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    print("Loading data...")

    for epoch in range(args.start_epoch + 1, EPOCHS):
        metric_counter.clear()
        models.scheduler_G.step(epoch)
        models.scheduler_D.step(epoch)
        for param_group in models.optimizer_G.param_groups:
            lr = param_group['lr']

        iteration = 0
        # --------------------train---------------------
        tq = tqdm(train_dataloader, total=len(train_dataloader))
        tq.set_description('Epoch {}, lr {:.6f}'.format(epoch, lr))
        for images in tq:
            iteration += 1
            loss = models.update_parameters(images, mode=int(iteration % STEP_TURN))
            metric_counter.add_losses(loss)
            tq.set_postfix(loss=metric_counter.loss_message())
        metric_counter.write_to_tensorboard(epoch, validation=False)

        # --------------------validation--------------------
        tq = tqdm(test_dataloader, total=len(test_dataloader))
        tq.set_description('Val_Epoch {}'.format(epoch))
        metric_counter.clear()
        iteration = 0
        with torch.no_grad():
            for images in tq:
                iteration += 1
                loss = models.Validation(images)
                metric_counter.add_losses(loss)
                tq.set_postfix(loss=metric_counter.loss_message())
            metric_counter.write_to_tensorboard(epoch, validation=True)

        if metric_counter.update_best_model():
            torch.save(models.netG.state_dict(), CHECKPOINT_DIR + 'best.pkl')
        torch.save(models.netG.state_dict(), CHECKPOINT_DIR + 'last.pkl')
        torch.save(models.netD.state_dict(), CHECKPOINT_DIR + 'netD.pkl')


if __name__ == '__main__':
    main()
