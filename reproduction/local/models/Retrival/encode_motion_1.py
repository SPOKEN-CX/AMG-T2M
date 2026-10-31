import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from src.config import read_config
from src.load import load_model_from_cfg
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from src.data.collate import collate_x_dict
import warnings
import time
import os
import random
from multiprocessing import Pool
from functools import partial

warnings.simplefilter("ignore", FutureWarning)
logger = logging.getLogger(__name__)

# ========== 全局变量 ==========
model = None
normalizer = None
device = 'cuda'


def encode_motion(motions: list, model, normalizer, device='cuda'):
    """
    将一组 motion（.npy 里的数组）编码为 latent 表示。
    motions: List[np.ndarray]，每个数组是一个 motion。
    """
    from src.model.tmr import get_score_matrix  # 根据实际需求导入

    # collate_x_dict 在 src.data.collate 下，需要根据实际工程结构做导入
    from src.data.collate import collate_x_dict

    motion_x_dict_list = []
    for motion_arr in motions:
        motion = torch.from_numpy(motion_arr).to(torch.float)
        motion = normalizer(motion)
        motion_x_dict_list.append({"x": motion.to(device), "length": len(motion)})

    with torch.inference_mode():
        motion_x_batch = collate_x_dict(motion_x_dict_list)
        lat_m = model.encode(motion_x_batch, sample_mean=False)

    return lat_m


def init_worker(run_dir, ckpt_name, device_):
    """
    每个子进程初始化时加载模型与相关变量。
    """
    global model
    global normalizer
    global device

    device = device_

    # 加载配置
    cfg = read_config(run_dir)

    # 设置随机种子
    seed_everything(cfg.seed)

    # 实例化 model
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    # 实例化 normalizer
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    print(f"进程初始化完成，使用设备: {device}")


def process_motion(npy_file, source_folder, output_folder):
    """
    对传入的 npy_file 进行编码，并将编码结果保存到 output_folder 中。
    """
    global model, normalizer, device

    # 组装完整的输入文件路径
    npy_path = os.path.join(source_folder, npy_file)
    if not os.path.isfile(npy_path):
        print(f"文件不存在: {npy_path}")
        return

    try:
        # 读取 npy 文件
        motion_arr = np.load(npy_path)

        # 编码
        lat_m = encode_motion([motion_arr], model, normalizer, device)
        lat_m_np = lat_m.cpu().numpy()  # 转为 numpy 数组

        # 输出文件与原文件同名，保存在目标文件夹
        out_path = os.path.join(output_folder, npy_file)
        np.save(out_path, lat_m_np)
        print(f"已处理并保存: {out_path}")

    except Exception as e:
        print(f"编码或保存时出现错误，文件: {npy_path}, 错误信息: {e}")


def main():
    # 配置路径（根据实际情况进行修改）
    run_dir = '/home/peng/PythonProjects/T2M-GPT-main/models/Retrival/models/tmr_humanml3d_guoh3dfeats'
    ckpt_name = 'last'
    device_ = 'cuda'  # 或 'cpu'

    # 源文件夹与目标文件夹（请根据实际需求修改）
    source_folder = "/home/peng/PythonProjects/Dataset/HumanML3D/new_joint_vecs"
    output_folder = "/home/peng/PythonProjects/Dataset/HumanML3D/motion_retrival"

    # 若目标文件夹不存在，则创建
    os.makedirs(output_folder, exist_ok=True)

    # 获取源文件夹中所有 .npy 文件
    npy_files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

    if not npy_files:
        print("源文件夹中没有找到 .npy 文件。")
        return

    # 创建进程池，指定两个进程
    with Pool(processes=2, initializer=init_worker, initargs=(run_dir, ckpt_name, device_)) as pool:
        # 使用 partial 固定 source_folder 和 output_folder 参数
        func = partial(process_motion, source_folder=source_folder, output_folder=output_folder)
        # 并行处理
        pool.map(func, npy_files)

    print("所有文件已处理完成。")


if __name__ == "__main__":
    main()
