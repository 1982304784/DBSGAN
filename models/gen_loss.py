import torch
import torch.nn as nn
import torchvision.models as models


class ContentLoss(object):
    def __init__(self, loss):
        """
        generator's loss func
        :param loss: L1 || L2 (nn.L1Loss() || nn.MSELoss())
        """
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


def contentFunc():
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    model = nn.Sequential()
    model = model.eval()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == conv_3_3_layer:
            break
    return model


class PerceptualLoss:

    def __init__(self, loss_mode, lamb_pixel=0.5, lamb_content=0.006):
        """
        :param loss_mode: L1 || L2 (nn.L1Loss() || nn.MSELoss())
        :param lamb_pixel: lambda_pixel
        :param lamb_content: lambda_content
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contentFunc = contentFunc().to(self.device)
        self.criterion = loss_mode
        self.lambda_pixel = lamb_pixel
        self.lambda_content = lamb_content
        for param in self.contentFunc.parameters():
            param.requires_grad = False

    def get_loss(self, fakeIm, realIm):
        fakeIm = fakeIm + 0.5
        realIm = realIm + 0.5
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm).detach()
        loss_content = self.criterion(f_fake, f_real)
        return self.lambda_content * loss_content, self.lambda_pixel * self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


def get_gen_loss(gen_loss_mode, lamb_1=0.5, lamb_2=0.006):
    """
    :param gen_loss_mode: perceptual || L1
    :param lamb_1: lambda_pixel  default 0.5
    :param lamb_2: lambda_content default 0.006
    :return: gen_loss
    """
    if gen_loss_mode == 'perceptual':
        content_loss = PerceptualLoss(nn.MSELoss(), lamb_pixel=lamb_1, lamb_content=lamb_2)
    elif gen_loss_mode == 'l1':
        content_loss = ContentLoss(nn.L1Loss())
    else:
        raise ValueError("ContentLoss [%s] not recognized." % gen_loss_mode)

    return content_loss
