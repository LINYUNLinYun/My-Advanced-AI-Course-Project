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
        self.scaler = torch.amp.GradScaler('cuda') # 混合精度训练
        
    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(loader, desc="  Training", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 开启混合精度，省显存+加速
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
        return running_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        running_iou = 0.0
        running_dice = 0.0
        pq_iou_sum = 0.0
        pq_tp = 0
        pq_fp = 0
        pq_fn = 0
        all_probs = []
        all_targets = []
        eps = 1e-7
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="  Evaluating", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                # Sigmoid + 阈值处理
                probs = outputs.sigmoid()
                pr_masks = (probs > 0.5).float()
                
                # 计算 IoU
                tp, fp, fn, tn = smp.metrics.get_stats(pr_masks.long(), masks.long(), mode='binary', threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                running_iou += iou
                running_dice += dice

                # PQ 统计（简单的单实例近似）
                for b in range(masks.size(0)):
                    gt_mask = masks[b] > 0.5
                    pred_mask = pr_masks[b] > 0.5
                    has_gt = gt_mask.any().item()
                    has_pred = pred_mask.any().item()

                    if has_gt and has_pred:
                        inter = (gt_mask & pred_mask).sum().item()
                        union = (gt_mask | pred_mask).sum().item()
                        iou_inst = inter / (union + eps)
                        if iou_inst >= 0.5:
                            pq_tp += 1
                            pq_iou_sum += iou_inst
                        else:
                            pq_fp += 1
                            pq_fn += 1
                    elif has_pred and not has_gt:
                        pq_fp += 1
                    elif has_gt and not has_pred:
                        pq_fn += 1

                # 收集概率用于 AP 计算
                all_probs.append(probs.detach().cpu())
                all_targets.append(masks.detach().cpu())

        # 处理 AP（像素级 PR AUC）
        probs_flat = torch.cat(all_probs).flatten()
        targets_flat = torch.cat(all_targets).flatten()
        ap = self._average_precision(probs_flat, targets_flat, eps)

        pq_den = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn + eps
        pq = pq_iou_sum / pq_den if (pq_tp + pq_fp + pq_fn) > 0 else 1.0

        num_batches = len(loader)
        return {
            "iou": (running_iou / num_batches).item(),
            "dice": (running_dice / num_batches).item(),
            "ap": ap,
            "pq": pq,
        }

    @staticmethod
    def _average_precision(probs, targets, eps=1e-7):
        """计算二分类 AP（像素级），避免额外依赖。"""
        if targets.sum() == 0:
            return 0.0

        scores, order = torch.sort(probs, descending=True)
        targets_sorted = targets[order]

        tp_cum = torch.cumsum(targets_sorted, dim=0)
        fp_cum = torch.cumsum(1 - targets_sorted, dim=0)

        recalls = tp_cum / (targets.sum() + eps)
        precisions = tp_cum / (tp_cum + fp_cum + eps)

        recalls = torch.cat([torch.tensor([0.0], device=recalls.device), recalls])
        precisions = torch.cat([torch.tensor([1.0], device=precisions.device), precisions])

        ap = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        return ap.item()