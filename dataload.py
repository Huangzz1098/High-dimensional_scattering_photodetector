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

        # load spectrum
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]                                 # Picture name
                spectrum = np.array(parts[1:], dtype=np.float32)    # Corresponding spectrum
                self.labels[filename] = spectrum

        self.L_size = img_size                                      # Cut size

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        image = Image.open(join(self.image_path, img_name))
        image = np.asarray(image, dtype=np.float32)

        # Acquire spectral data
        img_id = img_name[:-4]  # Remove file extension
        if img_id in self.labels:
            spectrum = self.labels[img_id]
        else:
            print(f"⚠️ 警告：未找到 {img_id} 的光谱数据！")
            spectrum = np.zeros(100, dtype=np.float32)  

        # Random cutting
        H, W = image.shape
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        image = image[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        # Normalization
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # 
        image_tensor = transforms.ToTensor()(image)
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32)

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

        # Read spectral data
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]                                 # 
                spectrum = np.array(parts[1:], dtype=np.float32)    # 
                self.labels[filename] = spectrum

        self.L_size = img_size                                      # 

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        image = Image.open(join(self.image_path, img_name))
        image = np.asarray(image, dtype=np.float32)

        # Acquire spectral data
        img_id = img_name[:-4]  # 
        if img_id in self.labels:
            spectrum = self.labels[img_id]
        else:
            print(f"⚠️ 警告：未找到 {img_id} 的光谱数据！")
            spectrum = np.zeros(100, dtype=np.float32)  # 

        # Random cutting
        H, W = image.shape
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        image = image[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        
        image_tensor = transforms.ToTensor()(image)
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32)

        return image_tensor, spectrum_tensor, img_name

    def __len__(self):
        return len(self.image_filenames)

