import os
import torch
import time
from src.dataset import get_dataloaders
from src.models import get_model
from src.trainer import Trainer
from src.utils import set_seed, count_parameters, plot_history, plot_predictions, save_logs

# ================= 配置区域 =================
CONFIG = {
    "DATA_DIR": "./dataset",
    "RESULTS_DIR": "./results",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_SIZE": 256,
    "BATCH_SIZE": 32,     # 8GB显存建议16
    "EPOCHS": 30,
    "LR": 0.0001,
    "SEED": 42,
    "MODELS": [
        "CBAM_UNET",
        "UNet_ResNet34",       # Baseline
        "UNet++_ResNet34",     # 架构复杂化
        "AttnUNet_ResNet34",   # 机制改进
        "TransUNet_MiT"        # Transformer (新颖点)
    ]
}
# ===========================================

def main():
    # 1. 初始化
    set_seed(CONFIG["SEED"])
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    print(f"🚀 开始实验，运行设备: {CONFIG['DEVICE']}")
    
    # 2. 准备数据
    train_loader, test_loader = get_dataloaders(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"], CONFIG["IMG_SIZE"])
    
    results = {}
    trained_models = {}
    
    # 3. 循环训练模型
    for model_name in CONFIG["MODELS"]:
        print(f"\n==========================================")
        print(f"正在处理模型: {model_name}")
        
        # 获取模型实例
        model = get_model(model_name)
        params_count = count_parameters(model)
        print(f"模型参数量: {params_count:.2f} M")
        
        # 训练管理器
        trainer = Trainer(model, CONFIG["DEVICE"], CONFIG["LR"])
        
        history = {"loss": [], "iou": []}
        start_time = time.time()
        
        # Training Loop
        for epoch in range(CONFIG["EPOCHS"]):
            train_loss = trainer.train_epoch(train_loader)
            val_iou = trainer.evaluate(test_loader)
            
            history["loss"].append(train_loss)
            history["iou"].append(val_iou.item())
            
            print(f"Ep {epoch+1}/{CONFIG['EPOCHS']} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f}")
            
        print(f"耗时: {(time.time()-start_time)/60:.1f} min")
        
        # 保存结果
        results[model_name] = history
        trained_models[model_name] = model
        
        # 获取现在的日期、时间作为保存文件名的一部分
        now_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # 保存权重
        torch.save(model.state_dict(), os.path.join(CONFIG["RESULTS_DIR"], f"{now_date_time}_{model_name}.pth"))
        
        # 释放显存
        torch.cuda.empty_cache()
        
    # 4. 生成报告
    print(f"\n📊 正在生成对比报告...")
    plot_history(results, CONFIG["RESULTS_DIR"],save_name=f"{now_date_time}_metrics_comparison.png")
    plot_predictions(trained_models, test_loader, CONFIG["DEVICE"], CONFIG["RESULTS_DIR"],save_name=f"{now_date_time}_visual_comparison.png")
    save_logs(results, CONFIG["RESULTS_DIR"],save_name=f"{now_date_time}_experiment_logs.csv")
    
    print(f"✅ 所有实验完成！结果已保存在 {CONFIG['RESULTS_DIR']}")

if __name__ == "__main__":
    main()