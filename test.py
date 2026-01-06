import os
import torch
import time


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    # import os

    # 获取现在的日期、时间作为保存文件名的一部分
    now_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"{now_date_time}.pth")
        
import os

# 修改这里为你的实际路径
ROOT_DIR = "./dataset/archive" 

img_dir = os.path.join(ROOT_DIR, "images")
label_dir = os.path.join(ROOT_DIR, "labels")

print(f"检查图片目录: {img_dir}")
print(f"检查标签目录: {label_dir}")

# 1. 检查目录是否存在
if not os.path.exists(img_dir) or not os.path.exists(label_dir):
    print("❌ 错误：目录不存在！请检查路径拼写。")
    exit()

# 2. 列出前 5 个文件
img_files = sorted(os.listdir(img_dir))[:5]
label_files = sorted(os.listdir(label_dir))[:5]

print("\n--- 图片文件夹的前5个文件 ---")
print(img_files)

print("\n--- 标签文件夹的前5个文件 ---")
print(label_files)

# 3. 尝试匹配测试
print("\n--- 尝试匹配 ---")
for img_name in img_files:
    # 假设的匹配逻辑 (原代码逻辑)
    assumed_mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(label_dir, assumed_mask_name)
    
    if os.path.exists(mask_path):
        print(f"✅ 成功匹配: {img_name} -> {assumed_mask_name}")
    else:
        print(f"❌ 匹配失败: {img_name} -> 找不到 {assumed_mask_name}")
        # 尝试看看是不是后缀名不一样，或者名字里多了东西
        # 比如 1001.jpg 对应的可能是 1001_label.png


if __name__ == "__main__":
    # main()
    pass
    

