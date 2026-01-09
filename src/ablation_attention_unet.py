import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Highly Configurable Attention U-Net
支持特性：
1. use_smart_ratio: 智能调整通道压缩比，防止浅层特征丢失。
2. use_residual: 开启残差连接 (Res-UNet模式)。
3. attention_mode: 控制注意力添加的位置 ('all', 'deep_only', 'none')。
"""

# ==========================================
# 1. 改进版 CBAM 模块
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, smart_ratio=False):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # === [改进方案一] Smart Ratio ===
        # 如果开启 smart_ratio，对于少于64通道的层，强制降低压缩倍率
        # 保证中间层至少有 8 个通道，避免信息瓶颈
        if smart_ratio:
            if in_planes < 64: 
                real_ratio = 4 
            elif in_planes < 128:
                real_ratio = 8
            else:
                real_ratio = ratio
        else:
            real_ratio = ratio
            
        mid_planes = max(4, in_planes // real_ratio)
        
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
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, smart_ratio=False):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio, smart_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# ==========================================
# 2. 改进版 卷积块 (支持残差)
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, 
                 use_cbam=False, use_residual=False, smart_ratio=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.use_cbam = use_cbam
        self.use_residual = use_residual

        # 基础卷积部分
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # CBAM 模块
        if self.use_cbam:
            self.cbam = CBAM(out_channels, smart_ratio=smart_ratio)

        # === [改进方案二] Residual Connection ===
        # 如果输入输出通道数不同，需要用 1x1 卷积调整 Shortcut 维度
        self.shortcut = nn.Sequential()
        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        # 1. 主干路径
        out = self.conv(x)
        
        # 2. 注意力机制
        if self.use_cbam:
            out = self.cbam(out)
        
        # 3. 残差连接 (ResNet思想: H(x) = F(x) + x)
        if self.use_residual:
            out = out + self.shortcut(x)
            # 注意：这里的 ReLU 一般在相加后不再做，因为 conv 里面已经 relu 过了
            # 如果想严格模仿 ResNet，可以在这里再加一个 F.relu(out)
            
        return out

# ==========================================
# 3. 辅助模块 (Down, Up)
# ==========================================

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, **kwargs) # 传递配置参数
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, **kwargs):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==========================================
# 4. 完整的 Configurable U-Net
# ==========================================

class ConfigurableUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_c=64, 
                 config=None):
        """
        config (dict): 控制实验变量的字典
            - use_smart_ratio (bool): 是否开启智能通道压缩
            - use_residual (bool): 是否开启残差连接
            - attention_mode (str): 'all' | 'deep_only' | 'none'
        """
        super(ConfigurableUNet, self).__init__()
        
        # 默认配置
        default_config = {
            'use_smart_ratio': False,
            'use_residual': False,
            'attention_mode': 'all' # 可选: 'all', 'deep_only', 'none'
        }
        if config:
            default_config.update(config)
        self.cfg = default_config
        
        print(f"🔄 模型初始化配置: {self.cfg}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # === [改进方案三] Attention Schedule ===
        # 定义哪些层开启 CBAM
        def get_cbam_flag(layer_name):
            mode = self.cfg['attention_mode']
            if mode == 'none':
                return False
            if mode == 'all':
                return True
            if mode == 'deep_only':
                # 定义深层：Down2, Down3, Down4, Up1, Up2
                # 浅层：Inc, Down1, Up3, Up4 (保留高分辨率特征不被抑制)
                if layer_name in ['down2', 'down3', 'down4', 'up1', 'up2']:
                    return True
                return False
            return False

        # 提取通用参数，简化代码
        common_args = {
            'use_residual': self.cfg['use_residual'],
            'smart_ratio': self.cfg['use_smart_ratio']
        }

        # ---- Encoder ----
        self.inc = DoubleConv(n_channels, base_c, 
                              use_cbam=get_cbam_flag('inc'), **common_args)
        
        self.down1 = Down(base_c, base_c * 2, 
                          use_cbam=get_cbam_flag('down1'), **common_args)
        
        self.down2 = Down(base_c * 2, base_c * 4, 
                          use_cbam=get_cbam_flag('down2'), **common_args)
        
        self.down3 = Down(base_c * 4, base_c * 8, 
                          use_cbam=get_cbam_flag('down3'), **common_args)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, 
                          use_cbam=get_cbam_flag('down4'), **common_args)

        # ---- Decoder ----
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, 
                      use_cbam=get_cbam_flag('up1'), **common_args)
        
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, 
                      use_cbam=get_cbam_flag('up2'), **common_args)
        
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, 
                      use_cbam=get_cbam_flag('up3'), **common_args)
        
        self.up4 = Up(base_c * 2, base_c, bilinear, 
                      use_cbam=get_cbam_flag('up4'), **common_args)
        
        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)

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
# 5. 控制变量实验示例
# ==========================================

if __name__ == '__main__':
    # 假设输入
    x = torch.randn(2, 3, 256, 256)
    
    print("----- 实验 1: 原始 Attention U-Net (你之前的版本) -----")
    model_v1 = ConfigurableUNet(3, 1, config={
        'use_smart_ratio': False,
        'use_residual': False,
        'attention_mode': 'all'
    })
    # out = model_v1(x) # 跑一下确保没报错
    
    print("\n----- 实验 2: 开启 Smart Ratio (方案一) -----")
    # 期望：缓解浅层特征丢失
    model_v2 = ConfigurableUNet(3, 1, config={
        'use_smart_ratio': True, 
        'use_residual': False,
        'attention_mode': 'all'
    })

    print("\n----- 实验 3: 开启 Residual Connection (方案二) -----")
    # 期望：训练更稳定，梯度更好传导
    model_v3 = ConfigurableUNet(3, 1, config={
        'use_smart_ratio': False,
        'use_residual': True, # 重点
        'attention_mode': 'all'
    })

    print("\n----- 实验 4: 只在深层加 Attention (方案三) -----")
    # 期望：保留浅层纹理，只在语义层做筛选，通常 IoU 最高
    model_v4 = ConfigurableUNet(3, 1, config={
        'use_smart_ratio': False,
        'use_residual': False,
        'attention_mode': 'deep_only' # 重点
    })
    
    print("\n----- 实验 5: 缝合怪 (全开 - 推荐) -----")
    # 结合了所有优点
    model_final = ConfigurableUNet(3, 1, config={
        'use_smart_ratio': True,
        'use_residual': True,
        'attention_mode': 'deep_only'
    })
    
    out = model_final(x)
    print(f"\n✅ 最终输出尺寸: {out.shape}")