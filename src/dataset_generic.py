import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GenericSegmentationDataset(Dataset):
    """
    通用数据集加载器，适用于：
    root/
      images/ (放原图)
      masks/  (放标签图，文件名要和原图一致，或者后缀不同)
    """
    def __init__(self, root, mode='train', img_size=256):
        self.root = root
        self.img_size = img_size
        self.mode = mode
        
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        
        self.files = sorted(os.listdir(self.img_dir))
        
        # 简单的 9:1 切分
        split_idx = int(len(self.files) * 0.9)
        if mode == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        # 假设 mask 文件名和 image 一样 (如果是 .jpg 对应 .png，这里要改后缀)
        # 比如: mask_name = file_name.replace('.jpg', '.png')
        mask_name = file_name 
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 转灰度

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 转 Tensor
        img = transforms.ToTensor()(img)
        mask = np.array(mask)
        
        # 二值化处理 (假设 mask 是黑白的，白色是目标)
        mask[mask > 128] = 1
        mask[mask <= 128] = 0
        
        mask = torch.from_numpy(mask).long().unsqueeze(0)

        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, mask.float()

    def __len__(self):
        return len(self.files)