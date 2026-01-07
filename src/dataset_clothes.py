import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, root, mode='train', img_size=256):
        """
        Args:
            root: 数据集根目录 (例如 'dataset/archive')
            mode: 'train' 或 'test'
            img_size: 图片大小
        """
        self.root = root
        self.img_size = img_size
        self.mode = mode
        
        # 定义图片和标签的文件夹路径
        # 根据你的截图，结构是 archive/images 和 archive/labels
        self.images_dir = os.path.join(root, 'images')
        # 优先使用原始 .mat 标签（类别索引更可靠），避免彩色可视化标注导致整图为前景。
        self.labels_dir = os.path.join(root, 'labels_raw/pixel_level_labels_mat')
        
        # 检查路径是否存在
        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"未找到 images 或 labels 文件夹，请检查路径: {self.images_dir}")

        # 获取所有图片文件名，并排序确保对应
        # 过滤出 jpg/png 文件
        all_img_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # 【关键修复】只保留有对应标注文件的图片
        self.img_files = []
        for img_name in all_img_files:
            mask_name = os.path.splitext(img_name)[0] + ".mat"
            mask_path = os.path.join(self.labels_dir, mask_name)
            if os.path.exists(mask_path):
                self.img_files.append(img_name)
        
        print(f"Found {len(self.img_files)} images with valid annotations (out of {len(all_img_files)} total images)")
        
        # 手动划分训练集和测试集 (这里按 9:1 划分)
        split_idx = int(len(self.img_files) * 0.9)
        if mode == 'train':
            self.img_files = self.img_files[:split_idx]
        else:
            self.img_files = self.img_files[split_idx:]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. 获取文件名
        img_name = self.img_files[idx]
        
        # 2. 构建完整路径
        img_path = os.path.join(self.images_dir, img_name)
        
        # 推断 mask 文件名（与图片同名，后缀为 .mat）
        mask_name = os.path.splitext(img_name)[0] + ".mat"
        mask_path = os.path.join(self.labels_dir, mask_name)
        
        # 3. 加载图片
        image = Image.open(img_path).convert("RGB")
        
        # 尝试加载 Mask，如果找不到文件，为了防止报错，给一个全黑的
        if os.path.exists(mask_path):
            mat = loadmat(mask_path)
            if 'groundtruth' not in mat:
                raise KeyError(f"MAT 文件中未找到 'groundtruth': {mask_path}")
            mask_np = mat['groundtruth']  # (H, W), uint8 类别索引
        else:
            # 这种情况不应该发生，因为已经在__init__中过滤了
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # 4. 调整大小 (模仿 PetDataset)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        # 最近邻缩放以保持离散标签
        mask_np = Image.fromarray(mask_np).resize((self.img_size, self.img_size), Image.NEAREST)
        mask_np = np.array(mask_np)

        # ==========================================
        # 核心修改：处理 Mask 标签为二分类
        # ==========================================
        # 原始 Clothing Dataset 中：0=背景，1~59=各种衣服/身体部位
        # 我们的目标：0=背景，1=前景(人+衣服)
        
        new_mask = np.zeros_like(mask_np, dtype=np.uint8)
        # 将所有非背景像素 (类别索引 > 0) 设为 1
        new_mask[mask_np > 0] = 1 
        
        # 5. 转换为 Tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image) # 会自动归一化到 [0, 1]
        
        # Mask 转为 LongTensor (整数型)，并增加一个维度变为 (1, H, W)
        mask = torch.from_numpy(new_mask).long().unsqueeze(0)
        
        # 6. 标准化 (使用 ImageNet 均值方差，这是常规操作)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        return image, mask.float()

def get_dataloaders(data_dir, batch_size, img_size=256):
    """
    Args:
        data_dir: 指向包含 images 和 labels 的父文件夹 (例如 dataset/archive)
    """
    train_ds = ClothingDataset(root=data_dir, mode="train", img_size=img_size)
    test_ds = ClothingDataset(root=data_dir, mode="test", img_size=img_size)
    
    # num_workers=2 对于单卡训练通常够用了
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# ==========================================
# 测试代码 (直接运行此文件可检查)
# ==========================================
if __name__ == '__main__':
    # 假设你的目录结构是: 当前文件夹/dataset/archive/images...
    # 请根据你实际解压的位置修改这里
    ROOT_DIR = "./dataset/archive" 
    
    if os.path.exists(ROOT_DIR):
        print(f"正在读取数据: {ROOT_DIR}")
        train_loader, test_loader = get_dataloaders(ROOT_DIR, batch_size=4)
        
        # 取一个 batch 看看
        images, masks = next(iter(train_loader))
        print(f"图片 Batch 形状: {images.shape}") # 应该是 [4, 3, 256, 256]
        print(f"Mask Batch 形状: {masks.shape}")   # 应该是 [4, 1, 256, 256]
        print(f"Mask 中的唯一值 (应只有0和1): {torch.unique(masks)}")
        print("✅ 数据加载成功！")
    else:
        print(f"❌ 路径不存在: {ROOT_DIR}，请修改脚本底部的 ROOT_DIR 变量。")