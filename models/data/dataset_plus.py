'''
Author: HDT
'''

import os
import sys
from joblib import Parallel, delayed
import numpy as np
import torch
import h5py
import open3d as o3d
import torch.utils.data as data
from torchvision import transforms
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
sys.path.append(FILE_DIR)
from pointcloud_transforms import PointCloudShuffle, ToTensor
from torch.utils.data import DataLoader




# import show3d_balls as show3d_balls
# def show_points(point_array, color_array=None, radius=3):
#     assert isinstance(point_array, list)
#     all_color = None
#     if color_array is not None:
#         assert len(point_array) == len(color_array)
#         all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
#         for i, c in enumerate(color_array):
#             all_color[i][:] = c
#         all_color = np.concatenate(all_color, axis=0)
#     all_points = np.concatenate(point_array, axis=0)
#     show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)


def collect_cycle_obj_sence_dir(data_dir, cycle_range, obj_name, scene_range):
    dirs = []
    for cycle_id in range(cycle_range[0],cycle_range[1]):
        # for obj_id in range(obj_name[0],obj_name[1]):
        for scene_id in range(scene_range[0],scene_range[1]):
            # dirs.append(os.path.join(data_dir, 'cycle_{:0>4}'.format(cycle_id), '{}_'.format(obj_id) + '{:0>3}.h5'.format(scene_id)))
            dirs.append(os.path.join(data_dir, 'cycle_{:0>4}'.format(cycle_id),  '{:0>3}.h5'.format(scene_id)))      
    return dirs
   

def load_dataset_by_cycle_layer(dir, mode='train', collect_names=False):
    num_point_in_h5 = 16384

    try:
        f = h5py.File(dir)
        point = f['data'][:].reshape(num_point_in_h5, 3)*1000
        label = f['labels'][:]
        trans_label = label[:,:3].reshape(num_point_in_h5, 3)*1000
        rot_label = label[:,3:12].reshape(num_point_in_h5, 3, 3)
        vs_label = label[:, 12].reshape(num_point_in_h5)
        
        dir  = dir[:-3]+'_point5.h5'
        dir_poiint = dir
        f = h5py.File(dir)
        z_label1 = f['rot_label_z1'][:].reshape(num_point_in_h5, 3)
        x_label1 = f['rot_label_x1'][:].reshape(num_point_in_h5, 3)
        x_label2 = f['rot_label_x2'][:].reshape(num_point_in_h5, 3)

        dataset={ 'data': point,
                'trans_label':trans_label,
                'rot_label': rot_label,
                'vs_label': vs_label,
                    
                'z_label1': z_label1,
                'x_label1': x_label1,
                'x_label2': x_label2,
            }
    except:
        print(f'{dir}')
        dir = '/home/hdt/Desktop/hdt/data/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/brick/train/cycle_0000/001.h5'
        f = h5py.File(dir)
        point = f['data'][:].reshape(num_point_in_h5, 3)*1000
        label = f['labels'][:]
        trans_label = label[:,:3].reshape(num_point_in_h5, 3)*1000
        rot_label = label[:,3:12].reshape(num_point_in_h5, 3, 3)
        vs_label = label[:, 12].reshape(num_point_in_h5)
        dir  = dir[:-3]+'_point5.h5'
        dir_poiint = dir
        f = h5py.File(dir)
        z_label1 = f['rot_label_z1'][:].reshape(num_point_in_h5, 3)
        x_label1 = f['rot_label_x1'][:].reshape(num_point_in_h5, 3)
        x_label2 = f['rot_label_x2'][:].reshape(num_point_in_h5, 3)
        dataset={ 'data': point,
                'trans_label':trans_label,
                'rot_label': rot_label,
                'vs_label': vs_label,    
                'z_label1': z_label1,
                'x_label1': x_label1,
                'x_label2': x_label2,
            }
    return dataset










class Parametric_Dataset(data.Dataset):
    def __init__(self, data_dir, DATA_OBJ_DIR,DATA_OBJ_DIR_TEMPLATE,DATA_OBJ_NAME,cycle_range, obj_name, scene_range, mode='train',
                 transforms=None, collect_names=False):
        self.mode = mode
        self.DATA_OBJ_DIR = DATA_OBJ_DIR
        self.DATA_OBJ_DIR_TEMPLATE = DATA_OBJ_DIR_TEMPLATE
        self.DATA_OBJ_NAME = DATA_OBJ_NAME
        self.transforms = transforms
        self.collect_names = collect_names
        self.dataset_dir = collect_cycle_obj_sence_dir(data_dir, cycle_range, obj_name, scene_range)
        data_dir_template_point_cloud = os.path.join(self.DATA_OBJ_DIR_TEMPLATE,self.DATA_OBJ_NAME+'.ply')
        pcd = o3d.io.read_point_cloud(data_dir_template_point_cloud)
        template_point_cloud = np.asarray(pcd.points)
        self.template_point_cloud = template_point_cloud
     

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):
        dataset = load_dataset_by_cycle_layer(self.dataset_dir[idx])
        cycle_temp = self.dataset_dir[idx].split('/')[-2]
        cycle_index = int(cycle_temp.split('_')[1])
        obj_and_scene_temp = self.dataset_dir[idx].split('/')[-1]
        scene_index = int(obj_and_scene_temp[0:3])
        name = [cycle_index,scene_index]
        sample = {
            'point_clouds': dataset['data'].copy().astype(np.float32),
            'rot_label': dataset['rot_label'].copy().astype(np.float32),
            'trans_label':dataset['trans_label'].copy().astype(np.float32),
            'vis_label': dataset['vs_label'].copy().astype(np.float32),
            'z_label1': dataset['z_label1'].copy().astype(np.float32),
            'x_label1': dataset['x_label1'].copy().astype(np.float32),
            'x_label2': dataset['x_label2'].copy().astype(np.float32),
        }
        if self.collect_names:
            sample['name'] = name
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample

