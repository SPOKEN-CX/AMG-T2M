import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
from pathlib import Path
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        if dataset_name == 't2m':
            self.data_root = str(Path(__file__).resolve().parents[2] / 'dataset' / 'HumanML3D')
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
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
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        if is_test:
            split_file = pjoin(self.data_root, 'test.txt' if dataset_name == 't2m' else 'long_test.txt')
        else:
            split_file = pjoin(self.data_root, 'val.txt')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24
        # min_motion_len = 64

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):

            try:
                flag = False
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if not np.isfinite(motion).all():
                    print(f'Skipping {name}: motion contains NaN or Inf')
                    continue

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        #print('line',line)
                        text_dict = {}
                        line_split = line.strip().split('#')

                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:

                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name




def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class st2m_Text2Motion_withpast_Dataset_evalV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length
        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        sent_len_list = []
        for name in tqdm(id_list):
            try:
                motion = []
                motion_length = []

                for i in range(self.opt.mul_data_size):
                    motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    length_i = len(motion_i)
                    gap = length_i % self.opt.unit_length

                    if i == 0:
                        motion_i = motion_i[gap:]
                    else:
                        motion_i = motion_i[:length_i - gap]

                    motion.append(motion_i)
                    motion_length.append(len(motion_i))

                text_data = []

                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_data.append(text_dict)

                # 计算句子长度用于排序
                tokens = text_data[0]['tokens'] + text_data[1]['tokens']
                sent_len = len(tokens) + 2  # 加上 'sos/OTHER' 和 'eos/OTHER'
                sent_len_list.append((name, sent_len))

                data_dict[name] = {'motion': motion,
                                   'length': motion_length,
                                   'text': text_data}

            except Exception as e:
                print(f"处理 {name} 时出错: {e}")
                pass

        # 根据句子长度降序排序
        sent_len_list.sort(key=lambda x: x[1], reverse=True)
        self.name_list = [item[0] for item in sent_len_list]
        self.sent_len_list = [item[1] for item in sent_len_list]
        self.data_dict = {name: data_dict[name] for name in self.name_list}

        self.mean = mean
        self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
        id_name = self.name_list[idx]
        data = self.data_dict[id_name]
        motion_list, m_length_list, text_list = data['motion'], data['length'], data['text']

        motion = np.concatenate(motion_list, axis=0)
        m_length = m_length_list[0] + m_length_list[1]
        caption = text_list[0]['caption'] + ' --> ' + text_list[1]['caption']
        tokens = text_list[0]['tokens'] + text_list[1]['tokens']

        if len(tokens) < self.opt.max_text_len * 2:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len * 2 + 2 - sent_len)
        else:
            # 截断
            tokens = tokens[:self.opt.max_text_len * 2]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = (motion - self.mean) / self.std
        motion = np.concatenate((motion,
                                 np.zeros((self.max_motion_length * 2 - m_length, motion.shape[1]))
                                 ), axis=0)

        # 处理单个动作和文本
        for i in range(self.opt.mul_data_size):
            motion_i = motion_list[i]
            m_length_i = m_length_list[i]
            text_data_i = text_list[i]
            caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
            caption_i_split = caption_i.split(' ')
            if len(caption_i_split) > self.opt.max_text_len + 2:
                caption_i = ' '.join(caption_i_split[:self.opt.max_text_len + 2])

            if len(tokens_i) < self.opt.max_text_len:
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
                tokens_i = tokens_i + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len_i)
            else:
                tokens_i = tokens_i[:self.opt.max_text_len]
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
            pos_one_hots_i = []
            word_embeddings_i = []
            for token in tokens_i:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_i.append(pos_oh[None, :])
                word_embeddings_i.append(word_emb[None, :])
            pos_one_hots_i = np.concatenate(pos_one_hots_i, axis=0)
            word_embeddings_i = np.concatenate(word_embeddings_i, axis=0)

            motion_i = (motion_i - self.mean) / self.std
            motion_i = np.concatenate((motion_i,
                                       np.zeros((self.max_motion_length - m_length_i, motion_i.shape[1]))
                                       ), axis=0)

            if i == 0:
                word_embeddings_0 = word_embeddings_i
                pos_one_hots_0 = pos_one_hots_i
                caption_0 = caption_i
                sent_len_0 = sent_len_i
                motion_0 = motion_i
                m_length_0 = m_length_i
                tokens_0 = tokens_i
            elif i == 1:
                word_embeddings_1 = word_embeddings_i
                pos_one_hots_1 = pos_one_hots_i
                caption_1 = caption_i
                sent_len_1 = sent_len_i
                motion_1 = motion_i
                m_length_1 = m_length_i
                tokens_1 = tokens_i

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name




# class st2m_Text2Motion_withpast_Dataset_evalV2(data.Dataset):
#     def __init__(self, opt, mean, std, split_file, w_vectorizer):
#         self.opt = opt
#         self.w_vectorizer = w_vectorizer
#         self.max_motion_length = opt.max_motion_length
#         joints_num = opt.joints_num
#
#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#
#         for name in tqdm(id_list):
#             try:
#                 motion = []
#                 motion_length = []
#
#                 for i in range(self.opt.mul_data_size):
#                     motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
#                     length_i = len(motion_i)
#                     gap = length_i % self.opt.unit_length
#
#                     if i == 0:
#                         motion_i = motion_i[gap:]
#                     else:
#                         motion_i = motion_i[:length_i - gap]
#
#                     motion.append(motion_i)
#                     motion_length.append(len(motion_i))
#
#                 text_data = []
#                 # print('pjoin(opt.text_dir, name + .txt)',pjoin(opt.text_dir, name + '.txt'))
#
#                 with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
#                     for line in f.readlines():
#                         # print('line',line)
#                         text_dict = {}
#                         line_split = line.strip().split('#')
#                         caption = line_split[0]
#                         tokens = line_split[1].split(' ')
#                         text_dict['caption'] = caption
#                         text_dict['tokens'] = tokens
#                         text_data.append(text_dict)
#                 # print('motion',motion)
#                 # print('motion_length',motion_length)
#                 # print('text_data',text_data)
#                 data_dict[name] = {'motion': motion,
#                                    'length': motion_length,
#                                    'text': text_data}
#
#
#             except Exception as e:
#                 # Some motion may not exist in KIT dataset
#                 print(f"Error processing {name}: {e}")
#                 pass
#
#         self.mean = mean
#         self.std = std
#         self.data_dict = data_dict
#         self.name_list = id_list
#
#         # print('len(self.data_dict)', len(self.data_dict))
#
#     def inv_transform(self, data):
#         return data * self.std + self.mean
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         idx = item
#         id_name = self.name_list[idx]
#         data = self.data_dict[self.name_list[idx]]
#         motion_list, m_length_list, text_list = data['motion'], data['length'], data['text']
#
#         motion = np.concatenate(motion_list, axis=0)
#         m_length = m_length_list[0] + m_length_list[1]
#         caption = text_list[0]['caption'] + ' --> ' + text_list[1]['caption']
#         tokens = text_list[0]['tokens'] + text_list[1]['tokens']
#
#         if len(tokens) < self.opt.max_text_len * 2:
#             tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#             sent_len = len(tokens)
#             tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len * 2 + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.opt.max_text_len * 2]
#             tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)
#
#         motion = (motion - self.mean) / self.std
#         motion = np.concatenate((motion,
#                                    np.zeros((self.max_motion_length * 2 - m_length, motion.shape[1]))
#                                    ), axis=0)
#
#         for i in range(self.opt.mul_data_size):
#             motion_i = motion_list[i]
#             m_length_i = m_length_list[i]
#             text_data_i = text_list[i]
#             caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
#             caption_i_split = caption_i.split(' ')
#             if len(caption_i_split) > self.opt.max_text_len + 2:
#                 caption_i = caption_i_split[0]
#                 for j in range(self.opt.max_text_len + 1):
#                     caption_i = caption_i + ' ' + caption_i_split[j + 1]
#
#             if len(tokens_i) < self.opt.max_text_len:
#                 # pad with "unk"
#                 tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
#                 sent_len_i = len(tokens_i)
#                 tokens_i = tokens_i + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len_i)
#             else:
#                 # crop
#                 tokens_i = tokens_i[:self.opt.max_text_len]
#                 tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
#                 sent_len_i = len(tokens_i)
#             pos_one_hots_i = []
#             word_embeddings_i = []
#             for token in tokens_i:
#                 word_emb, pos_oh = self.w_vectorizer[token]
#                 pos_one_hots_i.append(pos_oh[None, :])
#                 word_embeddings_i.append(word_emb[None, :])
#             pos_one_hots_i = np.concatenate(pos_one_hots_i, axis=0)
#             word_embeddings_i = np.concatenate(word_embeddings_i, axis=0)
#
#
#             motion_i = (motion_i - self.mean) / self.std
#             motion_i = np.concatenate((motion_i,
#                                        np.zeros((self.max_motion_length - m_length_i, motion_i.shape[1]))
#                                        ), axis=0)
#
#             # if i == 0:
#             #     word_embeddings_0 = word_embeddings_i
#             #     pos_one_hots_0 = pos_one_hots_i
#             #     caption_0 = caption_i
#             #     sent_len_0 = sent_len_i
#             #     motion_0 = motion_i
#             #     m_length_0 = m_length_i
#             #     tokens_0 = tokens_i
#             # elif i == 1:
#             #     word_embeddings_1 = word_embeddings_i
#             #     pos_one_hots_1 = pos_one_hots_i
#             #     caption_1 = caption_i
#             #     sent_len_1 = sent_len_i
#             #     motion_1 = motion_i
#             #     m_length_1 = m_length_i
#             #     tokens_1 = tokens_i
#         # print('word_embeddings_0: ',word_embeddings_0,word_embeddings_0.shape)
#         # print('word_embeddings_1: ',word_embeddings_1,word_embeddings_1.shape)
#         # print('word_embeddings: ',word_embeddings,word_embeddings.shape)
#         # print('pos_one_hots_0: ',pos_one_hots_0,pos_one_hots_0.shape)
#         # print('pos_one_hots_1: ',pos_one_hots_1,pos_one_hots_1.shape)
#         # print('pos_one_hots: ',pos_one_hots,pos_one_hots.shape)
#         # print('caption_0: ',caption_0)
#         # print('caption_1: ',caption_1)
#         # print('caption: ',caption)
#         # print('sent_len_0:',sent_len_0)
#         # print('sent_len_1:',sent_len_1)
#         # print('sent_len:',sent_len)
#         # print('motion_0:',motion_0,motion_0.shape)
#         # print('motion_1:',motion_1,motion_1.shape)
#         # print('motion:',motion,motion.shape)
#         # print('m_length_0:',m_length_0)
#         # print('m_length_1:',m_length_1)
#         # print('m_length:',m_length)
#         # print('tokens_0: ','_'.join(tokens_0))
#         # print('tokens_1:','_'.join(tokens_1))
#         # print('tokens:','_'.join(tokens))
#         # print('debug')
#         print('sent_len',sent_len)
#         return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name
#
#         # return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
#         #        sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
#         #        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name
