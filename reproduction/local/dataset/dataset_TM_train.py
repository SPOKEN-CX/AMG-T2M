import json

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import os


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
# class Text2MotionDataset(data.Dataset):
#     def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None):
#
#         self.max_length = 64
#         self.pointer = 0
#         self.dataset_name = dataset_name
#
#         self.unit_length = unit_length
#         # self.mot_start_idx = codebook_size
#         self.mot_end_idx = codebook_size
#         self.mot_pad_idx = codebook_size + 1
#         min_motion_len=40
#         if dataset_name == 't2m':
#             self.data_root = '../Dataset/HumanML3D'
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 22
#             radius = 4
#             fps = 20
#             self.max_motion_length = 26 if unit_length == 8 else 51
#             dim_pose = 263
#             kinematic_chain = paramUtil.t2m_kinematic_chain
#             split_file = pjoin(self.data_root, 'train.txt')
#         elif dataset_name == 'kit':
#             self.data_root = './dataset/KIT-ML'
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 21
#             radius = 240 * 8
#             fps = 12.5
#             dim_pose = 251
#             self.max_motion_length = 26 if unit_length == 8 else 51
#             kinematic_chain = paramUtil.kit_kinematic_chain
#         elif dataset_name == 'BABEL-TEACH':
#             self.data_root = '../Human-Datasets/BABEL-TEACH'
#             self.motion_dir = pjoin(self.data_root, 'BABEL_TEACH_joint_vecs_2')
#             self.raw_text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts')
#             self.text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts_GPT-processed')
#             self.joints_num = 22
#             self.mul_data_size=2
#             radius = 4
#             fps = 20
#             self.max_motion_length = 196
#             dim_pose = 263
#             split_file = pjoin(self.data_root, 'train_12103.txt')
#             val_split_file = pjoin(self.data_root, 'val_2163.txt')
#
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#
#         new_name_list = []
#         data_dict = {}
#         for name in tqdm(id_list):
#             try:
#                 #print(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))
#                 m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))
#                 # Read text
#                 with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
#                     text_data = []
#                     flag = False
#                     lines = f.readlines()
#
#                     for line in lines:
#                         try:
#                             text_dict = {}
#                             line_split = line.strip().split('#')
#                             caption = line_split[0]
#                             t_tokens = line_split[1].split(' ')
#                             f_tag = float(line_split[2])
#                             to_tag = float(line_split[3])
#                             f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                             to_tag = 0.0 if np.isnan(to_tag) else to_tag
#
#                             text_dict['caption'] = caption
#                             text_dict['tokens'] = t_tokens
#                             if f_tag == 0.0 and to_tag == 0.0:
#                                 flag = True
#                                 text_data.append(text_dict)
#                             else:
#                                 m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
#
#                                 if len(m_token_list_new) == 0:
#                                     continue
#                                 new_name = '%s_%f_%f'%(name, f_tag, to_tag)
#
#                                 data_dict[new_name] = {'m_token_list': m_token_list_new,
#                                                        'text':[text_dict]}
#                                 new_name_list.append(new_name)
#                         except:
#                             pass
#
#                 if flag:
#                     data_dict[name] = {'m_token_list': m_token_list,
#                                        'text':text_data}
#                     new_name_list.append(name)
#             except:
#                 pass
#         self.data_dict = data_dict
#         self.name_list = new_name_list
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         data = self.data_dict[self.name_list[item]]
#         m_token_list, text_list = data['m_token_list'], data['text']
#         print('m_token_list', len(m_token_list),'text', len(text_list))
#         m_tokens = random.choice(m_token_list)
#
#         text_data = random.choice(text_list)
#         caption= text_data['caption']
#
#
#         coin = np.random.choice([False, False, True])
#         # print(len(m_tokens))
#         if coin:
#             # drop one token at the head or tail
#             coin2 = np.random.choice([True, False])
#             if coin2:
#                 m_tokens = m_tokens[:-1]
#             else:
#                 m_tokens = m_tokens[1:]
#         m_tokens_len = m_tokens.shape[0]
#
#         if m_tokens_len+1 < self.max_motion_length:
#             m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
#         else:
#             m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)
#         return caption, m_tokens.reshape(-1), m_tokens_len

class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, unit_length=4, codebook_size=1024, tokenizer_name=None):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        min_motion_len = 40
        if dataset_name == 't2m':
            self.data_root = '../Dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            split_file = pjoin(self.data_root, 'train.txt')
            self.steps_text_dir = pjoin(self.data_root, 'texts_GPT_processed')
        elif dataset_name == 'kit':
            self.data_root = '../Dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            split_file = pjoin(self.data_root, 'train.txt')
            self.steps_text_dir = pjoin(self.data_root, 'steps_texts')
        elif dataset_name == 'BABEL-TEACH':
            self.data_root = '../Dataset/BABEL-TEACH'
            self.motion_dir = pjoin(self.data_root, 'BABEL_TEACH_joint_vecs_2')
            self.raw_text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts')
            self.text_dir = pjoin(self.data_root, 'BABEL_TEACH_texts_GPT-processed')
            self.joints_num = 22
            self.mul_data_size = 2
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            split_file = pjoin(self.data_root, 'train_12103.txt')
            val_split_file = pjoin(self.data_root, 'val_2163.txt')
            self.steps_text_dir = pjoin(self.data_root, 'BABEL_TEACH_steps_texts')
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, f'{name}.npy'))
                steps_file = pjoin(self.steps_text_dir, f'{name}.txt')
                if not os.path.exists(steps_file):
                    print(f'steps file {steps_file} does not exist')
                    continue
                steps_info_list = []
                with cs.open(steps_file, 'r', 'utf-8') as f_json:
                    for line in f_json:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            steps_dict = json.loads(line)
                            total_steps = len(steps_dict)
                            if total_steps != 3:
                                print(f'lines:{line}')
                            steps_info_list.append({'steps_dict': steps_dict, 'total_steps': total_steps})
                        except:
                            pass
                with cs.open(pjoin(self.text_dir, f'{name}.txt'), 'r') as f_txt:
                    lines = f_txt.readlines()
                text_data = []
                flag = False
                for line in lines:
                    try:
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        if np.isnan(f_tag):
                            f_tag = 0.0
                        if np.isnan(to_tag):
                            to_tag = 0.0
                        text_dict = {'caption': caption, 'tokens': t_tokens}
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            start_idx = int(f_tag * fps / unit_length)
                            end_idx = int(to_tag * fps / unit_length)
                            m_token_list_new = [tokens[start_idx:end_idx] for tokens in m_token_list if start_idx < end_idx]
                            if len(m_token_list_new) == 0:
                                continue
                            new_name = f'{name}_{f_tag}_{to_tag}'
                            data_dict[new_name] = {'m_token_list': m_token_list_new, 'text': [text_dict],'steps_info': steps_info_list}
                            #data_dict[sample_name]['steps_info'] = steps_info_list
                            new_name_list.append(new_name)
                    except:
                        pass
                if flag:
                    data_dict[name] = {'m_token_list': m_token_list, 'text': text_data, 'steps_info': steps_info_list}
                    new_name_list.append(name)
            except:
                pass
        # for sample_name in new_name_list:

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        sample_name = self.name_list[item]
        data = self.data_dict[sample_name]
        m_token_list = data['m_token_list']
        text_list = data['text']
        steps_info_list = data.get('steps_info', [])
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        if len(steps_info_list) > 0:
            random_steps_data = random.choice(steps_info_list)
            total_steps = random_steps_data['total_steps']
            steps_dict = random_steps_data['steps_dict']
        else:
            total_steps = 0
            steps_dict = {}

        # 随机丢弃 token（保持你原有逻辑）
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]
        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1), dtype=int) * self.mot_end_idx,
                np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int) * self.mot_pad_idx
            ], axis=0)
        else:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1), dtype=int) * self.mot_end_idx
            ], axis=0)

        # 把 steps_dict 直接转成字符串，避免 collate_fn 合并字典
        steps_dict_str = json.dumps(steps_dict, ensure_ascii=False)
        if total_steps !=3:
            print(f'line{caption} has {total_steps} steps,name is {sample_name}')
       #print(f'caption: {caption}')            #caption: a man stands with his right arm outstretched and moves his right leg in various directions
        #print(f'steps_dict_str: {steps_dict_str}') #steps_dict_str: {"Step 1": "Grasping the railing", "Step 2": "Ascsequently the stairs", "Step 3": "Maintain balance while ascending"}
        return caption, m_tokens.reshape(-1), m_tokens_len, total_steps, steps_dict_str

def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


