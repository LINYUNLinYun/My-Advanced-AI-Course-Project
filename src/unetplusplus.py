import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U-Net++ (Nested U-Net) Implementation
适用于：作为高阶对比模型 (Comparison Method)
特点：
1. 采用了密集的跳跃连接 (Dense Skip Pathways)。
2. 保留了 padding=1 和自动对齐逻辑，确保兼容你的数据集。
3. 为了方便训练，默认关闭了深监督 (Deep Supervision)，只输出最终结果，
   这样你可以直接替换 train.py 里的模型而不用改 Loss 函数。
"""

# ==========================================
# 1. 基础卷积块 (沿用之前的标准结构)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
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
# 2. U-Net++ 主体架构
# ==========================================
class NestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=32, deep_supervision=False):
        """
        Args:
            n_channels: 输入通道数 (3 for RGB)
            n_classes: 输出类别数
            base_c: 基础通道数 (建议设为32，因为U-Net++参数量大，设64容易爆显存)
            deep_supervision: 是否开启深监督。为了期末作业简单，默认False。
        """
        super(NestedUNet, self).__init__()
        
        self.deep_supervision = deep_supervision
        self.bilinear = True # 默认使用双线性插值上采样

        nb_filter = [base_c, base_c*2, base_c*4, base_c*8, base_c*16]

        # ---- 骨干网络 (Backbone) / 第一列 (L0) ----
        self.conv0_0 = DoubleConv(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], nb_filter[4])

        # ---- 嵌套层 / 第二列 (L1) ----
        # 输入来自于：左边一层的输出 + 下面一层的上采样
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        # ---- 嵌套层 / 第三列 (L2) ----
        # 输入来自于：左边两层的输出 (Dense) + 下面一层的上采样
        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2])

        # ---- 嵌套层 / 第四列 (L3) ----
        self.conv0_3 = DoubleConv(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1])

        # ---- 嵌套层 / 第五列 (L4) - 输出层 ----
        self.conv0_4 = DoubleConv(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # ---- 最终输出卷积 ----
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    # 这是一个辅助函数，用来做上采样并自动裁剪/填充，防止尺寸报错
    def _upsample_add(self, x, y):
        """
        x: 需要上采样的特征图 (来自深层)
        y: 需要拼接的特征图 (来自浅层)
        """
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([y, x], dim=1)
    
    # U-Net++ 的密集拼接逻辑比较复杂，这里我写得更通用一点
    def _upsample_cat(self, x_big, x_list):
        """
        x_big: 下一层来的特征 (需要上采样)
        x_list: 同一层左边来的所有特征列表 (需要拼接)
        """
        # 1. 上采样
        # 获取目标尺寸 (以列表里第一个特征图为准)
        target_h, target_w = x_list[0].size()[2], x_list[0].size()[3]
        x_big = F.interpolate(x_big, size=(target_h, target_w), mode='bilinear', align_corners=True)
        
        # 2. 拼接所有特征
        # 列表解包：把 x_list 里的所有 tensor 和上采样后的 x_big 拼在一起
        # 这里的维度对齐逻辑已经在 interpolate 里解决了
        return torch.cat(x_list + [x_big], dim=1)


    def forward(self, x):
        # -- 第 0 列 --
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))

        # -- 第 1 列 --
        x0_1 = self.conv0_1(self._upsample_cat(x1_0, [x0_0]))
        x1_1 = self.conv1_1(self._upsample_cat(x2_0, [x1_0]))
        x2_1 = self.conv2_1(self._upsample_cat(x3_0, [x2_0]))
        x3_1 = self.conv3_1(self._upsample_cat(x4_0, [x3_0]))

        # -- 第 2 列 --
        x0_2 = self.conv0_2(self._upsample_cat(x1_1, [x0_0, x0_1]))
        x1_2 = self.conv1_2(self._upsample_cat(x2_1, [x1_0, x1_1]))
        x2_2 = self.conv2_2(self._upsample_cat(x3_1, [x2_0, x2_1]))

        # -- 第 3 列 --
        x0_3 = self.conv0_3(self._upsample_cat(x1_2, [x0_0, x0_1, x0_2]))
        x1_3 = self.conv1_3(self._upsample_cat(x2_2, [x1_0, x1_1, x1_2]))

        # -- 第 4 列 --
        x0_4 = self.conv0_4(self._upsample_cat(x1_3, [x0_0, x0_1, x0_2, x0_3]))

        # -- 输出 --
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)

# ==========================================
# 3. 测试代码
# ==========================================
if __name__ == '__main__':
    # 模拟输入
    input_tensor = torch.randn(2, 3, 256, 256)
    
    # 实例化 U-Net++
    # 显存不够的话，一定要把 base_c 设小！U-Net++ 比 U-Net 占显存多很多。
    model = NestedUNet(n_channels=3, n_classes=1, base_c=32)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net++ 参数量: {total_params / 1e6:.2f} M")

    output = model(input_tensor)
    print(f"输出尺寸: {output.shape}")
    
    assert output.shape == (2, 1, 256, 256)
    print("✅ U-Net++ 测试通过！")