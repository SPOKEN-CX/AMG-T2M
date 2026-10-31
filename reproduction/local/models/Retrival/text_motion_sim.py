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

warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)


def text_motion_sim( texts: list, motions: list,  device: str = 'cuda'):
    """
    对多个文本与多个motion进行相似度计算，并返回相似度矩阵。

    参数：
    - run_dir: 模型所在的运行目录
    - texts: 文本列表，每个元素为字符串
    - motions: 已加载的motion数据的numpy数组列表，每个元素是一个numpy数组(形状一般为[T, nfeats])
    - ckpt_name: 模型检查点名称
    - device: 设备（'cuda'或'cpu'）
    """
    # 读取配置文件
    run_dir='/path/to/AMG-T2M/reproduction/local/models/Retrival/models/tmr_humanml3d_guoh3dfeats'
    ckpt_name='last'
    cfg = read_config(run_dir)
    seed_everything(cfg.seed)

    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    # 准备 motion 批量输入
    motion_x_dict_list = []
    for motion_arr in motions:
        motion = torch.from_numpy(motion_arr).to(torch.float)
        motion = normalizer(motion)
        motion_x_dict_list.append({"x": motion.to(device), "length": len(motion)})

    with torch.inference_mode():
        # motion -> latent (批量)
        motion_x_batch = collate_x_dict(motion_x_dict_list)
        # 将 sample_mean 设为 False，以便对多个 motion 的 batch 返回 [num_motions, latent_dim]
        lat_m = model.encode(motion_x_batch, sample_mean=False)

        # text -> latent (批量文本)
        text_x_dict = collate_x_dict(text_model(texts))
        lat_t = model.encode(text_x_dict, sample_mean=False)  # 返回 [B, latent_dim]

        # 计算相似度矩阵: lat_t [B, D], lat_m [N, D], 返回 [B, N]
        score_matrix = get_score_matrix(lat_t, lat_m).cpu().numpy()  # shape: [B, N]

    # 保留3位小数
    score_matrix = np.round(score_matrix, 3)

    logger.info(f"The similarity scores (0 <= s <= 1) are: \n{score_matrix}")
    return score_matrix


def text_motion_sim_V1(texts: list, motions: list, text_model, model, normalizer, device='cuda'):

    motion_x_dict_list = []
    for motion_arr in motions:
        motion = torch.from_numpy(motion_arr).to(torch.float)
        motion = normalizer(motion)
        motion_x_dict_list.append({"x": motion.to(device), "length": len(motion)})

    with torch.inference_mode():
        motion_x_batch = collate_x_dict(motion_x_dict_list)
        lat_m = model.encode(motion_x_batch, sample_mean=False)

        text_x_dict = collate_x_dict(text_model(texts))
        lat_t = model.encode(text_x_dict, sample_mean=False)

        score_matrix = get_score_matrix(lat_t, lat_m).cpu().numpy()

    score_matrix = np.round(score_matrix, 3)
    return score_matrix


# 使用示例（假设您在函数外部已经通过 np.load 获得 motions 数组列表）
import time


def text_motion_sim_V1(texts: list, motions: list, text_model, model, normalizer, device='cuda'):
    """
    计算 Text 和 Motion 之间相似度矩阵的函数，加入了时间统计。
    """
    # 1. 准备动作数据
    t_start_normalize = time.time()
    motion_x_dict_list = []
    for motion_arr in motions:
        motion = torch.from_numpy(motion_arr).to(torch.float)
        motion = normalizer(motion)  # 动作正则化
        motion_x_dict_list.append({"x": motion.to(device), "length": len(motion)})
    t_end_normalize = time.time()

    # 2. 编码：Motion -> latent
    t_start_motion_encode = time.time()
    with torch.inference_mode():
        motion_x_batch = collate_x_dict(motion_x_dict_list)
        lat_m = model.encode(motion_x_batch, sample_mean=False)
    t_end_motion_encode = time.time()

    # 3. 编码：Text -> latent
    t_start_text_encode = time.time()
    with torch.inference_mode():
        text_x_dict = collate_x_dict(text_model(texts))
        lat_t = model.encode(text_x_dict, sample_mean=False)
    t_end_text_encode = time.time()

    # 4. 计算相似度矩阵
    t_start_similarity = time.time()
    score_matrix = get_score_matrix(lat_t, lat_m).cpu().numpy()
    score_matrix = np.round(score_matrix, 3)
    t_end_similarity = time.time()

    # 打印各个阶段耗时
    print("------ Time Cost Breakdown ------")
    print(f"1) Normalize motions:      {t_end_normalize - t_start_normalize:.4f} s")
    print(f"2) Encode motions:         {t_end_motion_encode - t_start_motion_encode:.4f} s")
    print(f"3) Encode texts:           {t_end_text_encode - t_start_text_encode:.4f} s")
    print(f"4) Compute similarity:     {t_end_similarity - t_start_similarity:.4f} s")
    print("---------------------------------")

    return score_matrix

def text_motion_sim_V1(text, motions: list , model, normalizer, device='cuda'):
    start=time.time()
    """
    计算 Text 和 Motion 之间相似度矩阵的函数，加入了时间统计。
    """
    # 1. 准备动作数据
    t_start_normalize = time.time()
    motion_x_dict_list = []
    for motion_arr in motions:
        motion = torch.from_numpy(motion_arr).to(torch.float)
        motion = normalizer(motion)  # 动作正则化
        motion_x_dict_list.append({"x": motion.to(device), "length": len(motion)})
    t_end_normalize = time.time()

    # 2. 编码：Motion -> latent
    t_start_motion_encode = time.time()
    with torch.inference_mode():
        motion_x_batch = collate_x_dict(motion_x_dict_list)
        lat_m = model.encode(motion_x_batch, sample_mean=False)
    t_end_motion_encode = time.time()

    # 4. 计算相似度矩阵
    t_start_similarity = time.time()
    score_matrix = get_score_matrix(text, lat_m).cpu().numpy()
    score_matrix = np.round(score_matrix, 3)
    t_end_similarity = time.time()
    end=time.time()
    print(f'time consume:{end-start:.4f} s')

    return score_matrix

def text_motion_sim_V3(text, motions):
    score_matrix = get_score_matrix(text, motions)  # 一个 GPU Tensor，假设内部已在 cuda 上


    score_matrix = torch.round(score_matrix * 1000) / 1000.0

    # 3. 返回的依旧是一个 GPU 上的 Tensor
    return score_matrix

def text_motion_sim_V4(text, motions ):
    score_matrix = get_score_matrix(text, motions).cpu().numpy()
    score_matrix = np.round(score_matrix, 3)
    return score_matrix


# 以下为使用示例
if __name__ == "__main__":
    # 在调用前手动加载 npy 数据
    motion1 = np.load('/path/to/Dataset/HumanML3D/motion_retrival/000000.npy')
    motion1=torch.Tensor(motion1).to("cuda")
    print('motion1',motion1.shape)
    motion2 = np.load('/path/to/Dataset/HumanML3D/motion_retrival/000002.npy')
    motion2=torch.Tensor(motion2).to("cuda")
    motion3 = np.load('/path/to/Dataset/HumanML3D/motion_retrival/000003.npy')
    motion3=torch.Tensor(motion3).to("cuda")
    motion4 = np.load('/path/to/Dataset/HumanML3D/motion_retrival/000005.npy')
    motion4=torch.Tensor(motion4).to("cuda")
    motions=torch.stack([motion1,motion2,motion3,motion4],dim=0).squeeze(1)
    print('motions',motions.shape)
    text1 = np.load('/path/to/Dataset/HumanML3D/texts_retrival/000000.npy')
    text1=torch.Tensor(text1).to("cuda")
    print('text1', text1.shape)
    text2 = np.load('/path/to/Dataset/HumanML3D/texts_retrival/000002.npy')
    text2 = torch.Tensor(text2).to("cuda")
    text3 = np.load('/path/to/Dataset/HumanML3D/texts_retrival/000003.npy')
    text3 = torch.Tensor(text3).to("cuda")
    text4 = np.load('/path/to/Dataset/HumanML3D/texts_retrival/000005.npy')
    text4 = torch.Tensor(text4).to("cuda")
    texts=torch.stack([text1,text2,text3,text4],dim=0).squeeze(1)
    print('texts',texts.shape)
    device = "cuda"
    # run_dir = '/path/to/AMG-T2M/reproduction/local/models/Retrival/models/tmr_humanml3d_guoh3dfeats'
    # ckpt_name = 'last'
    # cfg = read_config(run_dir)
    # seed_everything(cfg.seed)

    # text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    # model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    # normalizer = instantiate(cfg.data.motion_loader.normalizer)
    scores_matrix=text_motion_sim_V1(texts, motions)
    print("Similarity Scores Matrix:")
    # scores_matrix = text_motion_sim_V1(
    #     texts=["A man spin around on right foot", "A man is standing", "A person walks backward"],
    #     motions=[motion1, motion2, motion3, motion4],
    #     text_model=text_model,
    #     model=model,
    #     normalizer=normalizer,
    #     device=device
    # )
    print(scores_matrix)

