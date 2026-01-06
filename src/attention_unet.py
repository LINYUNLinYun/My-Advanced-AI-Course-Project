import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention U-Net Implementation with CBAM
适用于：图像分割任务（语义分割/医学影像分割）
特点：集成CBAM注意力机制，支持Padding=1以保持尺寸，包含自动对齐逻辑防止尺寸报错。
"""

# ==========================================
# 1. CBAM 注意力模块 (即插即用的小积木)
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 为了节省参数，两个池化共用一个MLP
        # 注意：如果通道数太少，ratio会导致中间层为0，这里做了保护
        mid_planes = max(1, in_planes // ratio)
        
        self.fc1 = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道维度：最大池化 + 平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x) # 通道注意力
        result = out * self.sa(out) # 空间注意力
        return result

# ==========================================
# 2. 基础卷积块 (Double Convolution)
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_cbam=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # 两次3*3卷积，Padding=1保持尺寸，等效于做一次5*5卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x

# ==========================================
# 3. 完整的 Attention U-Net 网络架构
# ==========================================

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_c=64):
        """
        Args:
            n_channels (int): 输入图片的通道数 (RGB=3, 灰度=1)
            n_classes (int): 输出类别的数量 (二分类=1, 多分类=N)
            bilinear (bool): 是否使用双线性插值上采样 (节省显存推荐True)
            base_c (int): 基础通道数。默认64。如果显存爆了，请改为32。
        """
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ---- 编码器 Encoder (下采样) ----
        # 每一层的通道数翻倍: 64 -> 128 -> 256 -> 512
        self.inc = DoubleConv(n_channels, base_c, use_cbam=True)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # ---- 解码器 Decoder (上采样) ----
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        
        # ---- 输出层 ----
        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        # x: [Batch, 3, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 上采样并融合特征
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

# ==========================================
# 4. 辅助模块 (Down, Up, OutConv)
# ==========================================

class Down(nn.Module):
    """最大池化后接DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_cbam=True) # 这里也用了CBAM
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_cbam=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_cbam=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 是上采样后的特征图
        # x2 是来自编码器对应的特征图 (Skip Connection)
        
        # [核心逻辑] 处理尺寸不匹配问题
        # 即使输入尺寸是奇数，这里也能自动padding对齐
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
# 5. 测试代码 (直接运行此文件检查)
# ==========================================
if __name__ == '__main__':
    # 模拟一张 RGB 图片: Batch=2, Channel=3, Height=256, Width=256
    input_tensor = torch.randn(2, 3, 256, 256)
    
    # 实例化模型 (二分类任务 n_classes=1)
    # 如果显存不够，把 base_c 改成 32
    model = AttentionUNet(n_channels=3, n_classes=1, base_c=64)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")

    # 前向传播测试
    output = model(input_tensor)
    
    print(f"输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    
    # 简单的断言检查
    assert output.shape == (2, 1, 256, 256), "输出尺寸不对，请检查代码！"
    print("✅ 测试通过！可以直接在训练脚本中调用了。")