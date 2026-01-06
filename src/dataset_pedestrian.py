import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PennFudanDataset(Dataset):
    def __init__(self, root, mode='train', img_size=256):
        self.root = root
        self.img_size = img_size
        self.mode = mode
        
        # 排序确保一一对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
        # 简单划分训练集和测试集 (前150张训练，后20张测试)
        if mode == 'train':
            self.imgs = self.imgs[:-20]
            self.masks = self.masks[:-20]
        else:
            self.imgs = self.imgs[-20:]
            self.masks = self.masks[-20:]

    def __getitem__(self, idx):
        # 加载路径
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # 这里的 mask 是 instance mask (1,2,3...)

        # 1. Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 2. 转 Numpy 处理标签
        mask = np.array(mask)
        # 原始数据里不同的行人是1, 2, 3... 我们只需要区分 "有人" vs "背景"
        mask[mask > 0] = 1 
        
        # 3. 转 Tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        mask = torch.from_numpy(mask).long().unsqueeze(0) # (1, H, W)

        # 4. Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        img = normalize(img)

        return img, mask.float()

    def __len__(self):
        return len(self.imgs)

# 调用示例
# data_dir 指向解压后的 PennFudanPed 文件夹
# train_loader = DataLoader(PennFudanDataset("PennFudanPed", mode='train'), batch_size=4, shuffle=True)