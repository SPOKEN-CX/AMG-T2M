import os
import numpy as np
import random
import torch
import visualization.plot_3d_global as plot_3d
import warnings
import shutil
warnings.filterwarnings('ignore')
# ==== 设置路径 ====
# npy_folder = '/home/peng/PythonProjects/T2M-GPT-main/generation/generation'  # 替换为你的npy文件夹路径
#
# # ==== 获取所有.npy文件 ====
# all_npy_files = [f for f in os.listdir(npy_folder) if f.endswith('pred.npy')]
#
# # ==== 随机抽取10个文件 ====
# selected_files = random.sample(all_npy_files, 10)
#
# # ==== 分别读取内容和文件名 ====
# selected_arrays = []
# selected_filenames = []
#
# for filename in selected_files:
#     file_path = os.path.join(npy_folder, filename)
#     array = np.load(file_path)
#     selected_arrays.append(array)
#     selected_filenames.append(filename)
#
# # mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
# # std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
# from utils.motion_process import recover_from_ric
# for i in range(len(selected_files)):
#     xyz_cpu = np.load(os.path.join(npy_folder, selected_filenames[i]))
#     #xyz_cpu = xyz_cpu.reshape(1, -1, 22, 3)
#     print(xyz_cpu.shape)
#     gif_filename = selected_filenames[i].replace('.npy', '.gif')
#     gif_path = os.path.join('./short_motions', gif_filename)
#     np.save(os.path.join('./short_motions/npy', selected_filenames[i]), xyz_cpu)
#     pose_vis = plot_3d.draw_to_batch(xyz_cpu, None, [gif_path])# ==== 打印结果（可选） ====
# print("选中的文件名：", selected_filenames)
# print("对应的数组数量：", len(selected_arrays))
all_gif_files = [f for f in os.listdir('./short_motions/gif') if f.endswith('pred.gif')]
npy_dir='./short_motions/npy'
source_npy_dir='./generation/generation'
source_text_dir='../Dataset/HumanML3D/texts_clean'
text_dir='./short_motions/text'
for gif_file in all_gif_files:
    name=gif_file.split('_')[0]
    raw_name=gif_file.split('.')[0]
    print(name)
    npy_path=os.path.join(source_npy_dir,raw_name+'.npy')
    text_path=os.path.join(source_text_dir,name+'.txt')
    if os.path.exists(npy_path):
        shutil.copy(npy_path,npy_dir)
    else:
        print(f'{npy_path} not found')
    # if os.path.exists(text_path):
    #     shutil.copy(text_path,text_dir)
    # else:
    #     print(f'{text_path} not found')


