import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, device, lr=0.0001):
        self.model = model.to(device)
        self.device = device
        self.criterion = smp.losses.DiceLoss(mode='binary')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler() # 混合精度训练
        
    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(loader, desc="  Training", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 开启混合精度，省显存+加速
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.scale(self.optimizer).step()
            self.scaler.update()
            
            running_loss += loss.item()
            
        return running_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="  Evaluating", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                # Sigmoid + 阈值处理
                pr_masks = (outputs.sigmoid() > 0.5).float()
                
                # 计算 IoU
                tp, fp, fn, tn = smp.metrics.get_stats(pr_masks.long(), masks.long(), mode='binary', threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                running_iou += iou
                
        return running_iou / len(loader)