import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name =='t2m' else 24
        
        if dataset_name == 't2m':
            self.data_root = '../Dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            split_file = pjoin(self.data_root, 'train.txt')

            #kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.kit_kinematic_chain

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
            self.mul_data_size=2
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            if dataset_name=='BABEL_TEACH':
                motion = []
                split_motion = []
                split_motion_length = []
                total_length_count = 0
                for i in range(self.mul_data_size):
                    # motion_file = pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy')
                    motion_i = np.load(pjoin(self.motion_dir, name + '_C%03d' % i + '.npy'))
                    # print('motion_i',motion_i)
                    # print(motion_i.shape)
                    length_i = len(motion_i)
                    # print('leng_i',length_i)
                    total_length_count += length_i
                # print(f"{name} - total_length_count: {total_length_count}")  # 调试信息

                if (total_length_count < min_motion_len or total_length_count >= 200):
                    continue
                if total_length_count < min_motion_len:
                    print('total_length_count', total_length_count)
                for i in range(self.opt.mul_data_size):
                    motion_i = np.load(pjoin(self.motion_dir, name + '_C%03d' % i + '.npy'))
                    length_i = len(motion_i)
                    split_motion_length.append(length_i)
                    split_motion.append(motion_i)
                    if i == 0:
                        total_motion = motion_i
                    else:
                        total_motion = np.concatenate([total_motion, motion_i], axis=0)
                motion.append(total_motion)
                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'split_motion': split_motion,
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            else:
                try:

                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 200):
                        continue

                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'name': name}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except:
                    # Some motion may not exist in KIT dataset
                    pass


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4) : 
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length),
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


