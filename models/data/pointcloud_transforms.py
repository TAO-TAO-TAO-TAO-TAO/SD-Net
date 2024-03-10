import numpy as np
import torch

class PointCloudShuffle(object):
    def __init__(self):
        self.num_point = 16384

    def __call__(self, sample):
        pt_idxs = np.arange(0, self.num_point)
        np.random.shuffle(pt_idxs)

        sample['point_clouds'] = sample['point_clouds'][pt_idxs]
        if 'rot_label' in sample:
            sample['rot_label'] = sample['rot_label'][pt_idxs]
        if 'trans_label' in sample:
            sample['trans_label'] = sample['trans_label'][pt_idxs]
        if 'vis_label' in sample:
            sample['vis_label'] = sample['vis_label'][pt_idxs]
        if 'z_label1' in sample:
            sample['z_label1'] = sample['z_label1'][pt_idxs]
        if 'x_label1' in sample:
            sample['x_label1'] = sample['x_label1'][pt_idxs]
        if 'x_label2' in sample:
            sample['x_label2'] = sample['x_label2'][pt_idxs]
        if 'y_label1' in sample:
            sample['y_label1'] = sample['y_label1'][pt_idxs]
        return sample

class PointCloudJitter(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        all_noise = np.random.standard_normal(sample['point_clouds'].shape) * self.scale
        sample['point_clouds'] = sample['point_clouds'] + all_noise
        sample['point_clouds'] = sample['point_clouds'].astype(np.float32)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        
        sample['point_clouds'] = torch.from_numpy(sample['point_clouds'])
        if 'rot_label' in sample:
            sample['rot_label'] = torch.from_numpy(sample['rot_label'])
        if 'trans_label' in sample:
            sample['trans_label'] = torch.from_numpy(sample['trans_label'])
        if 'vis_label' in sample:
            sample['vis_label'] = torch.from_numpy(sample['vis_label'])

        return sample



    
