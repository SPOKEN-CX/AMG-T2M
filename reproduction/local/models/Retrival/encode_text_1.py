import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from src.config import read_config
from src.load import load_model_from_cfg
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from src.data.collate import collate_x_dict
from src.model.tmr import get_score_matrix
import warnings
import time

warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)
import os
import random
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial






# 您提供的 encode_text 函数
def encode_text(texts: list, text_model, model, device='cuda'):
    with torch.inference_mode():
        text_x_dict = collate_x_dict(text_model(texts))
        lat_t = model.encode(text_x_dict, sample_mean=False)
    return lat_t


# 全局变量用于存储模型
text_model = None
model = None
device = 'cuda'


def init_worker(run_dir, ckpt_name, device_):
    """
    初始化每个进程时加载模型。
    """
    global text_model
    global model
    global device
    device = device_

    # 加载配置
    cfg = read_config(run_dir)

    # 设置随机种子
    seed_everything(cfg.seed)

    # 实例化 text_model 和 model
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    print(f"进程初始化完成，使用设备: {device}")


def process_file(file_name, source_folder, output_folder):
    """
    处理单个文件：随机选择一行，编码并保存为.npy文件。
    """
    try:
        # 构建文件的完整路径
        file_path = os.path.join(source_folder, file_name)

        # 读取所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print(f"警告: 文件 {file_name} 是空的。")
            return

        # 随机选择一行
        selected_line = random.choice(lines).strip()

        # 编码选中的行
        lat_t = encode_text([selected_line], text_model, model, device)

        # 将张量转换为numpy数组并保存
        vector = lat_t.cpu().numpy()
        output_file = os.path.splitext(file_name)[0] + '.npy'
        output_path = os.path.join(output_folder, output_file)
        np.save(output_path, vector)
        print(f"已处理: {file_name}")
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")


def main():
    # 配置路径
    run_dir = '/path/to/AMG-T2M/reproduction/local/models/Retrival/models/tmr_humanml3d_guoh3dfeats'  # 您的 run_dir
    ckpt_name = 'last'  # 您的 checkpoint 名称
    device = 'cuda'  # 或 'cpu'，根据需要选择

    source_folder = "/path/to/Dataset/HumanML3D/texts_no_word_des"  # 替换为您的源文件夹路径
    output_folder = "/path/to/Dataset/HumanML3D/texts_retrival"  # 替换为您的目标文件夹路径

    # 创建目标文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 列出所有 .txt 文件
    txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

    if not txt_files:
        print("源文件夹中没有找到 .txt 文件。")
        return

    # 创建进程池，指定两个进程
    with Pool(processes=2, initializer=init_worker, initargs=(run_dir, ckpt_name, device)) as pool:
        # 使用 partial 固定 source_folder 和 output_folder 参数
        func = partial(process_file, source_folder=source_folder, output_folder=output_folder)
        pool.map(func, txt_files)

    print("所有文件已处理完成。")


if __name__ == "__main__":
    main()
