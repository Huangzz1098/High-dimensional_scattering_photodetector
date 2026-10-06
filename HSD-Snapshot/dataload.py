from os import listdir
from os.path import join
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import zoom

matplotlib.use('TkAgg')

class DatasetFromFolder_train(Dataset):
    def __init__(self, image_dir, label_file, img_size):
        super(DatasetFromFolder_train, self).__init__()

        self.image_path = image_dir
        self.image_filenames = sorted(listdir(self.image_path), key=lambda x: str(x[:-4]))

        # 读取光谱数据
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]                                 # 图片文件名（不含后缀）
                spectrum = np.array(parts[1:], dtype=np.float32)    # 对应光谱数据
                self.labels[filename] = spectrum

        self.L_size = img_size                                      # 训练时裁剪尺寸

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        image = Image.open(join(self.image_path, img_name))
        image = np.asarray(image, dtype=np.float32)

        # 获取光谱数据
        img_id = img_name[:-4]  # 去掉扩展名
        if img_id in self.labels:
            spectrum = self.labels[img_id]
        else:
            print(f"⚠️ 警告：未找到 {img_id} 的光谱数据！")
            raise KeyError(
                f"未找到图片 {img_name} 对应的光谱标签"
            )

        # 随机裁剪
        H, W = image.shape
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        image = image[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        # 归一化
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # 转 PyTorch 张量
        image_tensor = transforms.ToTensor()(image)
        pixel_num = 64 * 64

        if spectrum.size % pixel_num != 0:
            raise ValueError(
                f"{img_name} 标签长度 {spectrum.size} "
                f"不能被 64×64 整除"
            )

        # 根据标签长度自动推算光谱维数
        out_nc = spectrum.size // pixel_num

        spectrum = spectrum.reshape(
            64,
            64,
            out_nc
        )

        # 转为网络需要的 [100, 64, 64]
        spectrum = np.transpose(spectrum, (2, 0, 1))

        spectrum_tensor = torch.tensor(
            spectrum.copy(),
            dtype=torch.float32
        )

        return image_tensor, spectrum_tensor

    def __len__(self):
        return len(self.image_filenames)

# Imshow images (draft)
# plt.figure()
# plt.imshow(a_0, cmap='gray')
#
# plt.figure()
# plt.imshow(b_0, cmap='gray')
# plt.show()

class DatasetFromFolder_test(Dataset):
    def __init__(self, image_dir, label_file, img_size):
        super(DatasetFromFolder_test, self).__init__()

        self.image_path = image_dir
        self.image_filenames = sorted(listdir(self.image_path), key=lambda x: str(x[:-4]))

        # 读取光谱数据
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]                                 # 图片文件名（不含后缀）
                spectrum = np.array(parts[1:], dtype=np.float32)    # 对应光谱数据
                self.labels[filename] = spectrum

        self.L_size = img_size                                      # 训练时裁剪尺寸

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        image = Image.open(join(self.image_path, img_name))
        image = np.asarray(image, dtype=np.float32)

        # 获取光谱数据
        img_id = img_name[:-4]  # 去掉扩展名
        if img_id in self.labels:
            spectrum = self.labels[img_id]
        else:
            print(f"⚠️ 警告：未找到 {img_id} 的光谱数据！")
            raise KeyError(
                f"未找到图片 {img_name} 对应的光谱标签"
            )

        # 随机裁剪
        H, W = image.shape
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        image = image[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        # 归一化
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # 转 PyTorch 张量
        image_tensor = transforms.ToTensor()(image)
        pixel_num = 64 * 64

        if spectrum.size % pixel_num != 0:
            raise ValueError(
                f"{img_name} 标签长度 {spectrum.size} "
                f"不能被 64×64 整除"
            )

        # 根据标签长度自动推算光谱维数
        out_nc = spectrum.size // pixel_num

        spectrum = spectrum.reshape(
            64,
            64,
            out_nc
        )

        # 转为网络需要的 [100, 64, 64]
        spectrum = np.transpose(spectrum, (2, 0, 1))

        spectrum_tensor = torch.tensor(
            spectrum.copy(),
            dtype=torch.float32
        )

        return image_tensor, spectrum_tensor, img_name

    def __len__(self):
        return len(self.image_filenames)

