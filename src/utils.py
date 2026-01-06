import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    """计算模型参数量 (Millions)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def plot_history(results, save_dir, save_name = "metrics_comparison.png"):
    """绘制训练 Loss 和 IoU 曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        plt.plot(hist["loss"], label=name)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # IoU
    plt.subplot(1, 2, 2)
    for name, hist in results.items():
        plt.plot(hist["iou"], label=name)
    plt.title("Validation IoU")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

def plot_predictions(models, loader, device, save_dir, num_samples=3,save_name="visual_comparison.png"):
    """可视化预测结果"""
    images, masks = next(iter(loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            preds[name] = (model(images).sigmoid() > 0.5).cpu()
    
    # 绘图配置
    n_models = len(models)
    cols = 2 + n_models # Input, GT, Models...
    fig, axes = plt.subplots(num_samples, cols, figsize=(3*cols, 3*num_samples))
    
    # 反归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(num_samples):
        # 1. Input Image
        img_vis = images[i].cpu() * std + mean
        img_vis = np.clip(img_vis.permute(1, 2, 0).numpy(), 0, 1)
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')
        
        # 2. Ground Truth
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap='gray')
        axes[i, 1].set_title("GT")
        axes[i, 1].axis('off')
        
        # 3. Model Predictions
        for j, (name, pred) in enumerate(preds.items()):
            axes[i, 2+j].imshow(pred[i, 0].numpy(), cmap='gray')
            axes[i, 2+j].set_title(name.split('_')[0], fontsize=9)
            axes[i, 2+j].axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

def save_logs(results, save_dir, save_name="experiment_logs.csv"):
    """保存 CSV 数据"""
    data = {}
    for name, hist in results.items():
        data[f"{name}_loss"] = hist["loss"]
        data[f"{name}_iou"] = hist["iou"]
    pd.DataFrame(data).to_csv(os.path.join(save_dir, save_name), index_label="epoch")