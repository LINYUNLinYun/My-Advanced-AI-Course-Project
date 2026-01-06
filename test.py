import os
import torch
import time


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    # import os

    # 获取现在的日期、时间作为保存文件名的一部分
    now_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"{now_date_time+"_"}.pth")
        
        
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()