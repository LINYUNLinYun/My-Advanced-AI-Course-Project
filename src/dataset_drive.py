import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DriveDataset(Dataset):
    def __init__(self, root, mode='train', img_size=256):
        """
        root: 指向 'training' 或 'test' 文件夹
        DRIVE 数据集通常分为 training 和 test 两个文件夹
        """
        self.root = root
        self.img_size = img_size
        
        # DRIVE 的图片在 'images' 下，标签在 '1st_manual' 下
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, '1st_manual')
        
        self.images = sorted(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # DRIVE 的标签文件名通常是: 21_training.tif -> 21_manual1.gif
        # 这里需要根据你下载的具体文件名格式微调
        mask_name = img_name.split('_')[0] + "_manual1.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # DRIVE 的 mask 通常是 gif 格式

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = transforms.ToTensor()(img)
        
        # 处理 Mask
        mask = np.array(mask)
        mask = mask / 255.0 # DRIVE 是 0 和 255，归一化到 0-1
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        
        mask = torch.from_numpy(mask).long().unsqueeze(0)

        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, mask.float()

    def __len__(self):
        return len(self.images)