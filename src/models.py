import segmentation_models_pytorch as smp

def get_model(model_name):
    """
    模型工厂函数
    支持的模型配置:
    1. UNet_ResNet34: 基准模型
    2. UNet++_ResNet34: 架构改进
    3. AttnUNet_ResNet34: 机制改进 (Attention)
    4. TransUNet_MiT: 骨干改进 (Transformer)
    """
    
    # 公共参数
    ENCODER_RESNET = "resnet34"
    ENCODER_TRANSFORMER = "mit_b0" # SegFormer B0 (轻量级)
    weights = "imagenet"
    
    if model_name == "UNet_ResNet34":
        return smp.Unet(
            encoder_name=ENCODER_RESNET, 
            encoder_weights=weights, 
            in_channels=3, classes=1
        )
    
    elif model_name == "UNet++_ResNet34":
        return smp.UnetPlusPlus(
            encoder_name=ENCODER_RESNET, 
            encoder_weights=weights, 
            in_channels=3, classes=1
        )
    
    elif model_name == "AttnUNet_ResNet34":
        # scSE 模块是一种轻量级的空间+通道注意力机制
        return smp.Unet(
            encoder_name=ENCODER_RESNET, 
            encoder_weights=weights, 
            decoder_attention_type="scse", 
            in_channels=3, classes=1
        )
        
    elif model_name == "TransUNet_MiT":
        # 使用 Mix Transformer (SegFormer) 作为编码器
        return smp.Unet(
            encoder_name=ENCODER_TRANSFORMER, 
            encoder_weights=weights, 
            in_channels=3, classes=1
        )

    elif model_name == "CBAM_UNET":
        from src.attention_unet import AttentionUNet
        return AttentionUNet(n_channels=3, n_classes=1, base_c=64)
    elif model_name == "ABLATION_CBAM_UNET":
        from src.ablation_attention_unet import ConfigurableUNet
        return ConfigurableUNet(3, 1, config={
                    'use_smart_ratio': True,
                    'use_residual': True,
                    'attention_mode': 'deep_only',
                })
    elif model_name == "UNET":
        from src.unet import UNet
        return UNet(n_channels=3, n_classes=1, base_c=64)
    elif model_name == "UNETPLUSPLUS":
        from src.unetplusplus import NestedUNet
        return NestedUNet(n_channels=3, n_classes=1, base_c=64)
    
    else:
        raise ValueError(f"未知的模型名称: {model_name}")