import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image

class PetDataset(OxfordIIITPet):
    def __init__(self, root, split, mode, img_size=256):
        super().__init__(root=root, split=split, target_types="segmentation", download=True)
        self.img_size = img_size
        self.mode = mode

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        
        # 调整大小
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        image = np.array(image)
        mask = np.array(mask)

        # 处理 Mask 标签: 原始数据中 1=前景, 2=背景, 3=边缘
        # 转换为二分类: 0=背景, 1=宠物
        mask[mask == 2] = 0
        mask[mask == 3] = 0
        mask[mask == 1] = 1 
        
        # 转换为 Tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = torch.from_numpy(mask).long().unsqueeze(0) # (1, H, W)
        
        # 标准化 (ImageNet 统计数据)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        return image, mask.float()

def get_dataloaders(data_dir, batch_size, img_size=256):
    train_ds = PetDataset(root=data_dir, split="trainval", mode="train", img_size=img_size)
    test_ds = PetDataset(root=data_dir, split="test", mode="test", img_size=img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader