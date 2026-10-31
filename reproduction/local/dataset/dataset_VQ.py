import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import json


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        # ========== 根据数据集名称设置路径等参数 ==========
        if dataset_name == 't2m':
            self.data_root = '../Dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            self.text_retrival_dir = pjoin(self.data_root, 'texts_retrival')
            self.motion_retrival_dir = pjoin(self.data_root, 'motion_retrival')
            split_file = pjoin(self.data_root, 'train.txt')

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            split_file = pjoin(self.data_root, 'train.txt')

        elif dataset_name == 'BABEL_TEACH':
            self.data_root = '../Dataset/BABEL_TEACH'
            self.raw_text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts')
            self.motion_dir = pjoin(self.data_root, 'BABEL_TEACH_joint_vecs_2')
            self.text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts_GPT-processed')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            split_file = pjoin(self.data_root, 'train_12103.txt')
            val_split_file = pjoin(self.data_root, 'val_2163.txt')
            self.mul_data_size = 2
            self.meta_dir = 'checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        # ========== 读取均值与标准差，用于后续归一化 ==========
        self.mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'std.npy'))

        # ========== 初始化一些存储变量 ==========
        self.data = []
        self.lengths = []
        self.text = []
        self.split_motion = []
        self.split_motion_len = []

        # data_dict 用于存储 “名字 -> 对应数据” 的映射
        data_dict = {}
        new_name_list = []

        # 读取训练集/验证集的切分列表
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # ========== 开始遍历所有 id_list ==========
        for name in tqdm(id_list):
            if dataset_name == 'BABEL_TEACH':
                try:
                    split_motion = []
                    split_motion_length = []
                    total_length_count = 0
                    flag = True
                    for i in range(self.mul_data_size):
                        motion_i = np.load(pjoin(self.motion_dir, name + '_C%03d' % i + '.npy'))
                        length_i = len(motion_i)
                        if length_i < self.window_size:
                            flag = False
                        total_length_count += length_i

                    # 筛掉长度不符合要求的
                    if (total_length_count < self.window_size or total_length_count >= 200):
                        continue
                    for i in range(self.mul_data_size):
                        motion_i = np.load(pjoin(self.motion_dir, name + '_C%03d' % i + '.npy'))
                        length_i = len(motion_i)
                        split_motion_length.append(length_i - self.window_size)
                        split_motion.append(motion_i)
                        if i == 0:
                            total_motion = motion_i
                        else:
                            total_motion = np.concatenate([total_motion, motion_i], axis=0)

                    # 读取文本
                    text_data = []
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        text_dict = json.load(f)
                        text_data.append({'Total Step': int(text_dict["Total Step"])})
                        for step in range(1, int(text_dict["Total Step"]) + 1):
                            step_key = f"Step {step}"
                            if step_key in text_dict:
                                caption = text_dict[step_key]
                                text_data.append({'caption': caption})
                        if "Total description" in text_dict:
                            total_caption = text_dict["Total description"]
                            text_data.append({'total_caption': total_caption})

                    self.lengths.append(total_motion.shape[0] - self.window_size)
                    self.data.append(total_motion)
                    self.text.append(text_data)
                    self.split_motion.append(split_motion)

                except:
                    # BABEL_TEACH 中某些 motion 可能异常或不存在
                    pass

            else:
                try:
                    # 读取 motion
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    # 读取检索向量
                    motion_retrival = np.load(pjoin(self.motion_retrival_dir, name + '.npy'))
                    text_retrival = np.load(pjoin(self.text_retrival_dir, name + '.npy'))

                    # 筛掉太短的 motion
                    if motion.shape[0] < self.window_size:
                        continue

                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.data.append(motion)

                    # 读取文本
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()
                        for line in lines:
                            try:
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict = {
                                    'caption': caption,
                                    'tokens': t_tokens
                                }

                                if f_tag == 0.0 and to_tag == 0.0:
                                    # 没有局部标签，属于全局描述
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    # 有局部标签，生成一个新 name
                                    new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                    data_dict[new_name] = {
                                        'motion': motion,
                                        'text': [text_dict],
                                        # 在这里存储 motion_retrival & text_retrival
                                        'motion_retrival': motion_retrival,
                                        'text_retrival': text_retrival
                                    }
                                    new_name_list.append(new_name)

                            except:
                                pass

                        # 有全局描述则存到以原 name 为 key 的数据里
                        if flag:
                            data_dict[name] = {
                                'motion': motion,
                                'text': text_data,
                                'motion_retrival': motion_retrival,
                                'text_retrival': text_retrival
                            }
                            new_name_list.append(name)

                except:
                    # KIT / T2M 中某些 motion 文件可能不存在或异常
                    pass

        # ========== 将构造好的 data_dict 和 name_list 保存到成员变量 ==========
        self.data_dict = data_dict
        self.name_list = new_name_list

        print(f"总数据数量: {len(self.data_dict)}")
        print(f"name_list 长度: {len(self.name_list)}")
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        """把标准化的数据还原回去"""
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        """可选：计算每个 sample 的采样概率，用于 WeightedRandomSampler 等场景"""
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # ========== BABEL_TEACH 数据集的取数逻辑 ==========
        if self.dataset_name == 'BABEL_TEACH':
            motion = self.data[item]
            text_data = self.text[item]
            caption = None

            # 尝试提取 "total_caption"
            for entry in text_data:
                if 'total_caption' in entry:
                    caption = entry['total_caption']
                    break

            # 随机截取一个 window
            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx:idx + self.window_size]

            # 标准化
            motion = (motion - self.mean) / self.std

            return motion, caption

        else:
            data = self.data_dict[self.name_list[item]]
            motion = data['motion']
            text_list = data['text']
            # 取出 motion_retrival / text_retrival
            motion_retrival = data['motion_retrival']
            text_retrival = data['text_retrival']

            # 随机选一个 caption
            text_data = random.choice(text_list)
            caption = text_data['caption']

            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx:idx + self.window_size]

            motion = (motion - self.mean) / self.std
            return motion, caption, motion_retrival, text_retrival



class VQMotionDataset_V1(data.Dataset):
    def __init__(self, dataset_name, window_size=20, unit_length=4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = '../Dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            split_file = pjoin(self.data_root, 'train.txt')

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            # 此处若需要split_file，可自己指定

        elif dataset_name == 'BABEL_TEACH':
            self.data_root = '../Dataset/BABEL_TEACH'
            self.raw_text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts')
            self.motion_dir = pjoin(self.data_root, 'BABEL_TEACH_joint_vecs_2')
            self.text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts_GPT-processed')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            split_file = pjoin(self.data_root, 'train_12103.txt')
            val_split_file = pjoin(self.data_root, 'val_2163.txt')
            self.mul_data_size = 2
            self.meta_dir = 'checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        self.data = []  # 存储每个动作片段
        self.text = []  # 存储对应的文本描述
        self.lengths = []  # 存储对应的动作长度
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            if self.dataset_name == 'BABEL_TEACH':
                try:
                    # 读取文本
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        text_dict = json.load(f)

                    total_steps = int(text_dict.get("Total Step", 0))
                    if total_steps < self.mul_data_size:
                        # 如果总步数小于设定的mul_data_size则跳过
                        continue

                    # 对于每个动作步骤分别处理
                    for i in range(self.mul_data_size):
                        motion_path = pjoin(self.motion_dir, name + '_C%03d' % i + '.npy')
                        # if not os.path.exists(motion_path):
                        #     continue
                        motion_i = np.load(motion_path)
                        length_i = len(motion_i)
                        # 如果该动作长度小于window_size则跳过
                        if length_i < self.window_size or length_i >= 200:
                            continue

                        step_key = f"Step {i+1}"
                        caption = text_dict.get(step_key, None)
                        if caption is None:
                            # 如果没有相应的文本描述则跳过
                            continue

                        # 保存该动作步骤的数据
                        self.data.append(motion_i)
                        self.text.append(caption)
                        self.lengths.append(length_i - self.window_size)
                except:
                    pass
            else:
                try:
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if motion.shape[0] < self.window_size:
                        continue
                    # 此处如果不需要文本，可根据需求修改
                    # 暂时保持不变
                    self.data.append(motion)
                    self.text.append("")  # 无文本可用时
                    self.lengths.append(motion.shape[0] - self.window_size)
                except:
                    pass

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        caption = self.text[item]
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx + self.window_size]
        # Z Normalization
        motion = (motion - self.mean) / self.std
        return motion, caption

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 20,
               unit_length = 4):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
