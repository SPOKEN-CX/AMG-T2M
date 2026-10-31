from datetime import datetime
import numpy as np
import torch

from get_opt import get_opt



from utils.motion_process import *
from utils import paramUtil
from visualization.plot_3d_global import *


mean = np.load('/home/peng/PythonProjects/Human-Datasets/HumanML3D/Mean.npy')
std = np.load('/home/peng/PythonProjects/Human-Datasets/HumanML3D/Std.npy')
# mean = np.load('/home/peng/PythonProjects/ST2M-main/dataset/STDM/Mean.npy')
# std = np.load('/home/peng/PythonProjects/ST2M-main/dataset/STDM/Std.npy')
data_name='teach-24-1_C001'
data_name0='teach-24-1_C000'
save_dir='./vis_output'+data_name+'full'+'.mp4'
# data0=np.load('/home/peng/PythonProjects/ST2M-main/dataset/BABEL_TEACH/BABEL_TEACH_joint_vecs_2/'+data_name0+'.npy')
# data1=np.load('/home/peng/PythonProjects/ST2M-main/dataset/BABEL_TEACH/BABEL_TEACH_joint_vecs_2/'+data_name+'.npy')
# data=np.concatenate((data0,data1))
data_pred=np.load('/home/peng/PythonProjects/T2M-GPT-main/output/TEST_GPT/000115_pred.npy')
data_ground=np.load('/home/peng/PythonProjects/T2M-GPT-main/output/TEST_GPT/000115_gt.npy')

print('len(data_pred):',data_pred.shape)
print('len(data_ground):',data_ground.shape)

caption=''
kinematic_chain=paramUtil.t2m_kinematic_chain
dataset_opt_path = '/home/peng/PythonProjects/T2M-GPT-main/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
#dataset_opt_path = '/home/peng/PythonProjects/ST2M-main/checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM_slerp/opt.txt'
device='cuda' if torch.cuda.is_available() else 'cpu'
wrapper_opt = get_opt(dataset_opt_path, device)
# joint = recover_from_ric(torch.from_numpy(data).float(), wrapper_opt.joints_num).numpy()
# plot_3d_motion(save_dir,kinematic_chain,joint,title=data_name)