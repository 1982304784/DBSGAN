import torch
import torch.nn as nn
import functools
from torchvision import transforms
from DMPHN124 import DMPHN124
from fpn_inception import FPNInception
from datasets import RealBlurDataset
from datasets import GoProDataset
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
import numpy as np
# from models.models2.DMPHN2 import get_Generator
from models.generator import get_Generator


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
    return image_numpy.astype(imtype)


def get_images(inp, output1, output2, target) -> (float, float, float, np.ndarray):
    inp = tensor2im(inp)
    fake1 = tensor2im(output1.data)
    fake2 = tensor2im(output2.data)
    real = tensor2im(target.data)
    vis_img = np.hstack((inp, fake1, fake2, real))
    return vis_img


def save_deblur_images(images, save_dir):       # 保存去模糊图像
    filename = './image_DBSGAN/' + save_dir
    print(filename)
    torchvision.utils.save_image(images, filename)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # model = DMPHN124()
    # model = get_Generator(True)
    model = get_Generator(attention=True)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load('0610_realblur_last.pkl', map_location=torch.device('cpu')))

    model = model.to(device)

    test_dataset = RealBlurDataset(
        # blur_image_files='./dataset/GOPRO/test_blur_file.txt',
        blur_image_files='./dataset/RealBlur/RealBlur_R_test_list.txt',
        # sharp_image_files='./dataset/GOPRO/test_sharp_file.txt',
        sharp_image_files='./dataset/RealBlur/RealBlur_R_test_list.txt',
        # root_dir='./dataset/GOPRO/',
        root_dir='./dataset/RealBlur/',
        transform=transforms.Compose([
            transforms.ToTensor()
            ]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for iteration, images in enumerate(test_dataloader):
        with torch.no_grad():
            inputs = Variable(images['blur_image']-0.5).cuda()
            image_dir = images['dir']
            print(image_dir)
            deblur_image = model(inputs)
            save_dir = image_dir[0].split('/')

            save_dir = save_dir[6] + '_' + save_dir[7] + '_' + save_dir[8]
            print(save_dir)
            save_deblur_images(deblur_image.data + 0.5, save_dir)


if __name__ == '__main__':
    main()
