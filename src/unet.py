import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Standard U-Net Implementation (Baseline)
适用于：作为对比实验的基准模型（Baseline）
特点：标准的 (Conv => BN => ReLU) * 2 结构，去除了所有注意力机制。
保留了Padding=1和自动对齐逻辑，确保能跑通同样的数据集，不会报错。
"""

# ==========================================
# 1. 基础卷积块 (Double Convolution)
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # 纯粹的卷积结构，没有 Attention
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 2. 辅助模块 (Down, Up, OutConv)
# ==========================================

class Down(nn.Module):
    """最大池化后接DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels) # 标准版 DoubleConv
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样 => 裁剪/填充 => 拼接 => DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，卷积层需要减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 是上采样后的特征图
        # x2 是来自编码器对应的特征图 (Skip Connection)
        
        # [核心逻辑] 处理尺寸不匹配问题 (保留此逻辑以防止报错)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 3. 完整的 Standard U-Net 网络架构
# ==========================================

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_c=64):
        """
        Args:
            n_channels (int): 输入图片的通道数 (RGB=3, 灰度=1)
            n_classes (int): 输出类别的数量
            bilinear (bool): 是否使用双线性插值上采样
            base_c (int): 基础通道数 (默认64)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ---- 编码器 Encoder ----
        # 结构与 AttentionUNet 完全一致，只是没有 attention
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # ---- 解码器 Decoder ----
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        
        # ---- 输出层 ----
        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

# ==========================================
# 4. 测试代码
# ==========================================
if __name__ == '__main__':
    # 模拟输入
    input_tensor = torch.randn(2, 3, 256, 256)
    
    # 实例化标准 U-Net
    model = UNet(n_channels=3, n_classes=1, base_c=64)
    
    # 打印参数量 (你可以对比一下这个数字和 AttentionUNet 的数字)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Standard U-Net 参数量: {total_params / 1e6:.2f} M")

    output = model(input_tensor)
    print(f"输出尺寸: {output.shape}")
    assert output.shape == (2, 1, 256, 256)
    print("✅ Standard U-Net 测试通过！")