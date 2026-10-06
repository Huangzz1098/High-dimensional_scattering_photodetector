import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from os import listdir
from os.path import join
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from math import log10
import datetime
from dataload import DatasetFromFolder_train, DatasetFromFolder_test
from utils import get_scheduler, SSIM, init_net

from network_unet import UNetRes

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="datasets_hsi_leaf2") # Datasets files name
parser.add_argument('--img_size', type=int, default=3840, help='input image size') # Input image size
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=1e-4, help='learning rate for generator')      # 1e-4
parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy: step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
# load pre-trained model:
parser.add_argument('--SR_model_path', type=str, default='checkpoints/OC_OAM_spectrum_bit16/netG_epoch_144', help='folder to load SR model')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

trainout = opt.dataset + ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('-----------------------------Using GPU-----------------------------')

checkpoint = 0

print('===> Loading datasets')
root_dir = './dataset/'
train_dataset_path = os.path.join(root_dir, opt.dataset, "train")
test_dataset_path = os.path.join(root_dir, opt.dataset, "test")
train_set = DatasetFromFolder_train(
    train_dataset_path,
    train_dataset_path + ".txt",
    opt.img_size
)

test_set = DatasetFromFolder_test(
    test_dataset_path,
    test_dataset_path + ".txt",
    opt.img_size
)
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

## generator
# Loading SR model
# generator = torch.load("{}.pth".format(opt.SR_model_path)).to(device)

generator = UNetRes(
    in_nc=1,
    out_nc=50
).to(device) # out_nc, the size of spectrum

generator = init_net(
    generator,
    init_type='kaiming',
    init_gain=0.02
)


for param in generator.m_tail.parameters():
    param.requires_grad = True


### loss function
L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()
sml1_loss = nn.SmoothL1Loss()
# adversarial_criterion = nn.BCEWithLogitsLoss()
ssim_criterion = SSIM(window_size=11)

generator = generator.to(device)
content_criterion = L2_loss.to(device)
L1_loss = L1_loss.to(device)
SML1_loss = sml1_loss.to(device)

# learning rate
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(0.9, 0.999))
scheduler_generator = get_scheduler(optim_generator, opt)


print('--------------------------------------Model training--------------------------------------')
# History
history = pd.DataFrame()
G_loss = []
VAL_mae = []
VAL_mse = []
VAL_psnr = []
start_time = datetime.datetime.now()

best_loss = 100.
for epoch in range(opt.epoch_count, opt.nEpochs):
    mean_pce_loss = 0.0
    mean_l1_loss = 0.0
    mean_l2_loss = 0.0
    mean_ssim_loss = 0.0
    mean_total_loss = 0.0
    count = len(training_data_loader)

    generator.train()
    with torch.set_grad_enabled(True):
        for i, (inputs, targets) in enumerate(training_data_loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = generator(inputs)
            generator.zero_grad()

            # --------- calculate loss ---------
            # # L1 loss
            # generator_l1_loss = L1_loss(high_res_fake, high_res_real) + L2_loss(high_res_fake, high_res_real)
            # mean_l1_loss += generator_l1_loss

            # L2 loss
            generator_l2_loss = L2_loss(outputs, targets)
            mean_l2_loss += generator_l2_loss

            # Total loss
            generator_total_loss = 1 * generator_l2_loss
            mean_total_loss += generator_total_loss
            generator_total_loss.backward()
            optim_generator.step()

            # elapsed_time = datetime.datetime.now() - start_time
            # print('\r[%d/%d][%d/%d] Generator_Loss (Total): %.8f time: %s'
            #       % (epoch+1, opt.nEpochs, i+1, len(training_data_loader), generator_total_loss.item(), elapsed_time))

    scheduler_generator.step()

    # 求平均损失，生成图像以查看效果
    generator.eval()
    with torch.set_grad_enabled(False):
        G_loss.append(mean_total_loss.detach().cpu().numpy() / count)
        elapsed_time = datetime.datetime.now() - start_time
        print("===> TRAIN: Epoch[{}]:  Loss_G: {:.6f}  time: {}".format(epoch + 1, G_loss[-1], elapsed_time))

        mean_mae_loss = 0.0
        mean_mse_loss = 0.0
        mean_psnr_loss = 0.0
        count = len(testing_data_loader)
        record = len(testing_data_loader)-1
        for batch_ind, batch in enumerate(testing_data_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = generator(inputs)

            mae_loss = L1_loss(outputs, targets.detach())
            mean_mae_loss += mae_loss

            mse_loss = content_criterion(outputs, targets.detach())
            mean_mse_loss += mse_loss

            psnr_loss = 10 * log10(1 / mae_loss.item())
            mean_psnr_loss += psnr_loss

            # 显示图像，方便观察
            inputs = inputs.cpu().numpy()
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()
            if batch_ind == len(testing_data_loader) - 1:
                #### ---------- batch size >= 2 ----------
                if not os.path.exists(os.path.join("results", trainout)):
                    os.makedirs(os.path.join("results", trainout))

                gen_imgs = [inputs, targets, outputs]
                titles = ['Input', 'GT', 'Pred']
                r = min(4, inputs.shape[0])
                c = 3

                fig, axs = plt.subplots(
                    r,
                    c,
                    figsize=(12, 3 * r),
                    squeeze=False
                )
                for i in range(r):
                    # 显示图像（Input）
                    axs[i, 0].imshow(gen_imgs[0][i], cmap='gray')
                    axs[i, 0].set_title(titles[0])
                    axs[i, 0].axis('off')
                    # 显示 GT 光谱（折线图）
                    axs[i, 1].plot(
                        gen_imgs[1][i, :, 32, 32],
                        color='blue'
                    )
                    axs[i, 1].set_title(titles[1])
                    axs[i, 1].set_xticks([])  # 去掉 x 轴刻度
                    axs[i, 1].set_yticks([])  # 去掉 y 轴刻度
                    # 显示 Pred 光谱（折线图）
                    axs[i, 2].plot(
                        gen_imgs[2][i, :, 32, 32],
                        color='red'
                    )
                    axs[i, 2].set_title(titles[2])
                    axs[i, 2].set_xticks([])
                    axs[i, 2].set_yticks([])
                fig.savefig('results/{}/{}.png'.format(trainout, epoch + 1))
                # plt.show()
                plt.close()

        VAL_mae.append(mean_mae_loss.detach().cpu().numpy() / count)
        VAL_mse.append(mean_mse_loss.detach().cpu().numpy() / count)
        VAL_psnr.append(mean_psnr_loss / count)
        elapsed_time = datetime.datetime.now() - start_time
        print("===> VAL:   Epoch[{}]:  MAE: {:.6f}  MSE: {:.6f}  PSNR: {:.6f}dB  time: {}".format(
            epoch + 1, VAL_mae[-1], VAL_mse[-1], VAL_psnr[-1], elapsed_time))

    # 把每个epoch的衡量结果输出并保存
    epoch_result = dict(epoch=epoch+1, G_loss=round(float(G_loss[-1]), 6),
                        avg_mae=round(float(VAL_mae[-1]), 6),
                        avg_mse=round(float(VAL_mse[-1]), 6),
                        avg_psnr=round(float(VAL_psnr[-1]), 6))
    history = history._append(epoch_result, ignore_index=True)
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(os.path.join("checkpoints", trainout)):
        os.mkdir(os.path.join("checkpoints", trainout))
    history_path = "checkpoints/{}/history.csv".format(trainout)
    history.to_csv(history_path, index=False)

    # save the best model
    if (VAL_mae[-1]) < best_loss:
        net_g_model_out_path = "checkpoints/{}/netG_epoch_{}.pth".format(trainout, epoch + 1)
        torch.save(generator, net_g_model_out_path)
        best_loss = VAL_mae[-1]
        print("Checkpoint saved to {}".format("checkpoint" + trainout))

    # Save model at intervals
    if (epoch + 1) % 5 == 0:
        # if not os.path.exists("checkpoints"):
        #     os.mkdir("checkpoints")
        # if not os.path.exists(os.path.join("checkpoints", trainout)):
        #     os.mkdir(os.path.join("checkpoints", trainout))
        # net_g_model_out_path = "checkpoints/{}/netG_epoch_{}.pth".format(trainout, epoch + 1)
        # # net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(trainout, epoch)
        # torch.save(generator, net_g_model_out_path)
        # # torch.save(net_d, net_d_model_out_path)
        # print("Checkpoint saved to {}".format("checkpoint" + trainout))

        # 画出loss图
        x_ = range(opt.epoch_count, epoch + 1)
        plt.plot(x_, G_loss, label='Generator Losses')
        plt.plot(x_, VAL_mae, color='b', label='Val Loss')
        plt.legend()
        loss_image_path = "checkpoints/{}/loss.png".format(trainout)
        plt.savefig(loss_image_path)
        plt.close()

print('---------------Training is finished!!!---------------')



