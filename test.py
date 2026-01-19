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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="filename", help='Dataset name')
parser.add_argument('--img_size', type=int, default=256, help='Input image size')
parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size')
parser.add_argument('--model', type=str, default="model_folder")
parser.add_argument('--model_name', type=str, default="model_name")
parser.add_argument('--Tag', type=int, default=1, help='flag for saving images')
parser.add_argument('--SavePath', type=str, default="Outputs")
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

print('===> Loading test dataset')
root_dir = './dataset/'
test_dataset_path = os.path.join(root_dir, opt.dataset, "test")
test_set = DatasetFromFolder_test(test_dataset_path, test_dataset_path + ".txt", opt.img_size)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

print(f'===> Loading model from {opt.model}')
model_path = "checkpoints/{}/{}.pth".format(opt.model, opt.model_name)
generator = torch.load(model_path, map_location=device)
generator.eval().to(device)

L1_loss = torch.nn.L1Loss().to(device)
L2_loss = torch.nn.MSELoss().to(device)
ssim_criterion = SSIM(window_size=11).to(device)

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

with torch.no_grad():
    for batch_ind, batch in enumerate(testing_data_loader):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        image_filenames = batch[2][0]  # Acquire file name
        outputs = generator(inputs)

        mae_loss = L1_loss(outputs, targets)
        mse_loss = L2_loss(outputs, targets)

        mse_val = mse_loss.item()
        if mse_val == 0:
            psnr_loss = float('inf') 
        else:
            psnr_loss = 10 * log10(1 / mse_val)

        # psnr_loss = 10 * log10(1 / mse_loss.item())
        mean_mae_loss += mae_loss.  item()
        mean_mse_loss += mse_loss.item()
        mean_psnr_loss += psnr_loss

        if opt.Tag == 1:
            inputs = inputs.cpu().numpy()
            inputs = np.transpose(inputs, (0, 2, 3, 1)) 
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            sample_name = image_filenames.split('.')[0]

            # Save image
            in_Img = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs) + 1e-8)
            in_Img = np.asarray(in_Img[0, :, :, 0] * 65535, dtype=np.uint16)
            in_Img = Image.fromarray(in_Img, mode='I;16')
            in_Img.save(os.path.join(save_path1, image_filenames))

            # Save GT spectrum（CSV）
            gt_spectrum = targets.flatten()
            gt_df = pd.DataFrame(gt_spectrum)
            gt_df.to_csv(os.path.join(save_path, f"{sample_name}_gt.csv"), index=False, header=False)

            # Save Pred spectrum（CSV）
            pred_spectrum = outputs.flatten()
            pred_df = pd.DataFrame(pred_spectrum)
            pred_df.to_csv(os.path.join(save_path, f"{sample_name}_pred.csv"), index=False, header=False)

            # GT vs Pred
            plt.figure(figsize=(8, 4))
            plt.plot(gt_spectrum, color='blue', label="GT")
            plt.plot(pred_spectrum, color='red', linestyle="--", label="Pred")
            plt.xlabel("Pixel Index")
            plt.ylabel("Intensity")
            plt.legend()
            plt.title(f"Spectrum Comparison: {sample_name}")
            plt.savefig(os.path.join(save_path, f"{sample_name}.png"))
            plt.close()


mean_mae_loss /= count
mean_mse_loss /= count
mean_psnr_loss /= count
print(f'===> Test Results: MAE: {mean_mae_loss:.6f}, MSE: {mean_mse_loss:.6f}, PSNR: {mean_psnr_loss:.6f} dB')
