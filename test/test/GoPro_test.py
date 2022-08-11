import numpy as np
import pandas as pd
from PIL import Image
from psnr_ssim import calculate_psnr as PSNR
from psnr_ssim import calculate_ssim as SSIM
from torch.utils.data import Dataset, DataLoader


class LoadDir(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, root_dir):
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_direction = self.root_dir + image_name[0] + '/' + image_name[1] + '/' + image_name[2] + '/' + image_name[3]
        sharp_direction = self.root_dir + image_name[0] + '/' + image_name[1] + '/' + 'sharp' + '/' + image_name[3]
        return {'blur_dir': blur_direction, 'sharp_dir': sharp_direction}


if __name__ == '__main__':
    test_dataset = LoadDir(
        blur_image_files='./dataset/GOPRO/test_blur_file.txt',
        sharp_image_files='./dataset/GOPRO/test_sharp_file.txt',
        root_dir='./dataset/GOPRO/'
    )
    ROOT_DIR = './dataset/GOPRO/'
    RESULT_NAME = 'GOPRO_DBSGAN' + '.csv'
    sum_ssim = 0
    sum_psnr = 0
    data = []
    list_dir = []

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for iteration, images in enumerate(test_dataloader):
        SHARP_DIR_PATH = images['sharp_dir'][0]
        BLUR_DIR_PATH = images['blur_dir'][0]
        DEBLUR_DIR_PATH = BLUR_DIR_PATH.split('/')
        DEBLUR_DIR_PATH = ROOT_DIR + DEBLUR_DIR_PATH[6] + '_' + DEBLUR_DIR_PATH[7] + '_' + DEBLUR_DIR_PATH[8]  # ??
        deblur_dir, sharp_dir = DEBLUR_DIR_PATH, SHARP_DIR_PATH

        deblur = Image.open(deblur_dir).convert('RGB')
        deblur = np.array(deblur).astype(np.uint8)
        sharp = Image.open(sharp_dir).convert('RGB')
        sharp = np.array(sharp).astype(np.uint8)

        psnr = PSNR(sharp, deblur, crop_border=0)
        ssim = SSIM(sharp, deblur, crop_border=0)
        sum_ssim += ssim
        sum_psnr += psnr
        data.append([psnr, ssim])
        list_dir.append([sharp_dir, deblur_dir])
        print(psnr, ssim)

    # print(sum_psnr / iteration, sum_ssim / iteration)
    print('end')
    data = pd.DataFrame(data, columns=['psnr', 'ssim'])
    list_dir = pd.DataFrame(list_dir, columns=['sharp', 'blur'])
    result = pd.concat([data, list_dir], axis=1)
    result.to_csv(RESULT_NAME)
