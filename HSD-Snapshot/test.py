from __future__ import print_function
import argparse
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from math import log10
from PIL import Image
from dataload import DatasetFromFolder_test
from utils import SSIM
from network_unet import UNetRes

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="datasets_hsi_leaf2", help='Dataset name')
parser.add_argument('--img_size', type=int, default=3840, help='Input image size')
parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size')
parser.add_argument('--model', type=str, default="datasets_hsi_leaf2")
parser.add_argument('--model_name', type=str, default="netG_epoch_1")
parser.add_argument('--Tag', type=int, default=1, help='flag for saving images')
parser.add_argument('--SavePath', type=str, default="Outputs")
opt = parser.parse_args()

# 设备选择
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 载入测试数据
print('===> Loading test dataset')
root_dir = './dataset/'
test_dataset_path = os.path.join(root_dir, opt.dataset, "test")
test_set = DatasetFromFolder_test(test_dataset_path, test_dataset_path + ".txt", opt.img_size)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

# 载入模型
print(f'===> Loading model from {opt.model}')
model_path = "checkpoints/{}/{}.pth".format(opt.model, opt.model_name)
generator = torch.load(model_path, map_location=device)
generator.eval().to(device)

# 评估指标
L1_loss = torch.nn.L1Loss().to(device)
L2_loss = torch.nn.MSELoss().to(device)
ssim_criterion = SSIM(window_size=11).to(device)

# 保存路径
save_path1 = os.path.join(opt.SavePath, opt.dataset, "Input")
save_path = os.path.join(opt.SavePath, opt.dataset)

os.makedirs(save_path1, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

print('===> Starting evaluation')
mean_mae_loss, mean_mse_loss, mean_psnr_loss = 0.0, 0.0, 0.0
count = len(testing_data_loader)

def save_image(image_array, path):
    image_array = np.asarray(image_array * 65535, dtype=np.uint16)
    image = Image.fromarray(image_array, mode='I;16')
    image.save(path)

def to_stokes_hwc(array):
    array = np.asarray(array)

    # remove batch dimension
    if array.ndim == 4:
        array = array[0]

    # [4, 64, 64] -> [64, 64, 4]
    if array.ndim == 3 and array.shape[0] == 4:
        array = np.transpose(array, (1, 2, 0))

    # already [64, 64, 4]
    if array.ndim == 3 and array.shape[-1] == 4:
        return array

    raise ValueError(f"Unexpected Stokes output shape: {array.shape}")

def to_hwc(array):
    array = np.asarray(array)

    if array.ndim == 4:
        array = array[0]

    if array.ndim == 3 and array.shape[1] == 64 and array.shape[2] == 64:
        array = np.transpose(array, (1, 2, 0))

    if array.ndim == 3 and array.shape[0] == 64 and array.shape[1] == 64:
        return array

    raise ValueError(f"Unexpected output shape: {array.shape}")

with torch.no_grad():
    for batch_ind, batch in enumerate(testing_data_loader):
        print(f"{batch_ind + 1}/{len(testing_data_loader)}", flush=True)
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        image_filenames = batch[2][0]  # 获取文件名
        outputs = generator(inputs)

        # 计算损失
        mae_loss = L1_loss(outputs, targets)
        mse_loss = L2_loss(outputs, targets)

        mse_val = mse_loss.item()
        if mse_val == 0:
            psnr_loss = float('inf')  # 理论上PSNR=∞，可以用100之类的替代
        else:
            psnr_loss = 10 * log10(1 / mse_val)

        # psnr_loss = 10 * log10(1 / mse_loss.item())
        mean_mae_loss += mae_loss.  item()
        mean_mse_loss += mse_loss.item()
        mean_psnr_loss += psnr_loss

        if opt.Tag == 1:
            inputs = inputs.cpu().numpy()
            inputs = np.transpose(inputs, (0, 2, 3, 1))  # 调整维度
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            sample_name = image_filenames.split('.')[0]

            # 保存输入图像
            in_Img = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs) + 1e-8)
            in_Img = np.asarray(in_Img[0, :, :, 0] * 65535, dtype=np.uint16)
            in_Img = Image.fromarray(in_Img, mode='I;16')
            in_Img.save(os.path.join(save_path1, image_filenames))

            # targets / outputs -> [64, 64, 4]
            gt_cube = to_hwc(targets)
            pred_cube = to_hwc(outputs)

            num_channels = gt_cube.shape[-1]

            columns = [f"band_{i + 1}" for i in range(num_channels)]

            gt_df = pd.DataFrame(gt_cube.reshape(-1, num_channels), columns=columns)
            pred_df = pd.DataFrame(pred_cube.reshape(-1, num_channels), columns=columns)

            gt_df.to_csv(os.path.join(save_path, f"{sample_name}_gt.csv"), index=False)
            pred_df.to_csv(os.path.join(save_path, f"{sample_name}_pred.csv"), index=False)

            # save s1 GT and Pred images
            band_idx = gt_cube.shape[-1] - 1

            gt_band = gt_cube[:, :, band_idx]
            pred_band = pred_cube[:, :, band_idx]

            plt.imsave(os.path.join(save_path, f"{sample_name}_gt_band{band_idx + 1}.png"), gt_band, cmap="gray")
            plt.imsave(os.path.join(save_path, f"{sample_name}_pred_band{band_idx + 1}.png"), pred_band, cmap="gray")


# 计算平均损失
mean_mae_loss /= count
mean_mse_loss /= count
mean_psnr_loss /= count
print(f'===> Test Results: MAE: {mean_mae_loss:.6f}, MSE: {mean_mse_loss:.6f}, PSNR: {mean_psnr_loss:.6f} dB')
