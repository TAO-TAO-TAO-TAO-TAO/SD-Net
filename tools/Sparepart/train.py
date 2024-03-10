""" 
train FFNet.
Author: HDT
"""


gpus_is = True 
gpus_is = False

if gpus_is:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
else:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    


import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
PROJECT_NAME = os.path.basename(FILE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.append(ROOT_DIR)
import math
from datetime import datetime
import h5py
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import pffnet, load_checkpoint, save_checkpoint
from models.utils.train_helper import BNMomentumScheduler, OptimizerLRScheduler, SimpleLogger
from models.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from models.data.dataset_plus import Parametric_Dataset



if gpus_is :
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='59882') 
        torch.distributed.init_process_group(backend="nccl",init_method=dist_init_method, world_size=world_size,rank=rank)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



BATCH_SIZE = 16
MAX_EPOCH = 300  
TRAIN_DATA_HOLD_EPOCH = 3  
EVAL_STAP = 10 
DISPLAY_BATCH_STEP = 100  
SAVE_STAP = 50
BASE_LEARNING_RATE = 0.001
LR_DECAY_RATE = 0.7
MIN_LR = 1e-6 
LR_DECAY_STEP = 80
LR_LAMBDA = lambda epoch: max(BASE_LEARNING_RATE * LR_DECAY_RATE**(int(epoch / LR_DECAY_STEP)), MIN_LR)
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MIN = 0.001
BN_DECAY_STEP = LR_DECAY_STEP
BN_DECAY_RATE = 0.5
BN_LAMBDA = lambda epoch: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(epoch / BN_DECAY_STEP)), BN_MOMENTUM_MIN)



def generate_list(start, end, step):
    result = []
    for i in range(start, end, step):
        result.append([i,i+step])
    return result

LOG_NAME = 'train'  

CHECKPOINT_PATH = None

DATASET_DIR = '/data/hdt/data/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/brick/train/'
DATA_OBJ_DIR = None
DATA_OBJ_DIR_TEMPLATE= '/data/hdt/data/ParametricDataset/iap/ply_1024/'     
DATA_OBJ_NAME = 'Sileanebrick'
TRAIN_CYCLE_RANGES = generate_list(0, 250, 25) + generate_list(500, 725, 25)
TRAIN_OBJ_RANGE = None                              
TRAIN_SCENE_RANGE = [1,151]  
TEST_CYCLE_RANGE =  [725, 750]
TEST_OBJ_RANGE = None
TEST_SCENE_RANGE = [1,151]  

log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME)
logger = SimpleLogger(log_dir, FILE_PATH)
SummaryWriter_log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME, "tensorboard")
if not os.path.exists(SummaryWriter_log_dir): os.mkdir(SummaryWriter_log_dir)



if gpus_is:
    net = pffnet(  True, True)
    net =  net.to(device)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    num_gpus = torch.cuda.device_count()
    if CHECKPOINT_PATH is not None:
        net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, None)
    else:
        start_epoch = 0
    if num_gpus > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    if dist.get_rank() == 0:
        writer = SummaryWriter(SummaryWriter_log_dir)   
else:
    net = pffnet( True, True)
    net.to(device)
    writer = SummaryWriter(SummaryWriter_log_dir)

    if CHECKPOINT_PATH is not None:
        net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, None)
    else:
        start_epoch = 0


optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=BN_LAMBDA, last_epoch=start_epoch-1)
lr_scheduler = OptimizerLRScheduler(optimizer, lr_lambda=LR_LAMBDA, last_epoch=start_epoch-1)


transforms = transforms.Compose(
    [
        PointCloudShuffle(),
        ToTensor()
    ]
)


if gpus_is:
    if dist.get_rank() == 0:
        print('Loading test dataset')
    test_dataset = Parametric_Dataset(DATASET_DIR, DATA_OBJ_DIR,DATA_OBJ_DIR_TEMPLATE,DATA_OBJ_NAME,TEST_CYCLE_RANGE, TEST_OBJ_RANGE, TEST_SCENE_RANGE, transforms=transforms)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=test_sampler)
    if dist.get_rank() == 0:
        print('Test dataset loaded, test point cloud size:', len(test_dataset.dataset_dir))
else:
    print('Loading test dataset')
    test_dataset = Parametric_Dataset(DATASET_DIR, DATA_OBJ_DIR,DATA_OBJ_DIR_TEMPLATE,DATA_OBJ_NAME,TEST_CYCLE_RANGE, TEST_OBJ_RANGE, TEST_SCENE_RANGE, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('Test dataset loaded, test point cloud size:', len(test_dataset.dataset_dir))
train_dataset = None





def train_one_epoch(loader,epoch):
    logger.reset_state_dict('train trans1', 'train vs','train z','train x')
    if gpus_is:
        if dist.get_rank() == 0:
            logger.log_string('----------------TRAIN STATUS---------------')
    else:
            logger.log_string('----------------TRAIN STATUS---------------')
    net.train() 
    for batch_idx, batch_samples in enumerate(loader):
        if batch_idx == 2:
            start_time = time.time()

        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['point_clouds'].shape)).float()
        input_points_with_noise = batch_samples['point_clouds'] + xyz_noise*0.5
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device),
            'z_label1': batch_samples['z_label1'].to(device),
            'x_label1': batch_samples['x_label1'].to(device),
            'x_label2': batch_samples['x_label2'].to(device),    
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(device),
            'labels': labels
        }

        optimizer.zero_grad()
        pred_results, losses = net(inputs)
        losses['total'].backward()
        optimizer.step()
        log_state_dict = {  'train trans1': losses['trans_head'].item(),
                            'train vs': losses['vis_head'].item(),
                            'train z': losses['rot_z'].item(),
                            'train x': losses['rot_x'].item(),
                            }
        logger.update_state_dict(log_state_dict)
        if gpus_is:
            if dist.get_rank() == 0:
                if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
                    print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
                    logger.print_state_dict(log=False)
                if batch_idx == 2:
                    t = time.time() - start_time
                    print('Successfully train one batchsize in %f seconds.' % (t))
        else:                                    
                if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
                    print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
                    logger.print_state_dict(log=False)
                if batch_idx == 2:
                    t = time.time() - start_time
                    print('Successfully train one batchsize in %f seconds.' % (t))
    if gpus_is:
        if dist.get_rank() == 0:
            print('Current batch/total batch num: %d/%d'%(len(loader),len(loader)))
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)
    else:
            print('Current batch/total batch num: %d/%d'%(len(loader),len(loader)))
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)

def eval_one_epoch(loader,epoch):
    logger.reset_state_dict('eval trans1', 'eval vs', 'eval z', 'eval x',)

        
    if gpus_is:
        if dist.get_rank() == 0:
            logger.log_string('----------------EVAL STATUS---------------')
    else:
            logger.log_string('----------------EVAL STATUS---------------')


    net.eval() 
    loss_sum = 0
    for batch_idx, batch_samples in enumerate(loader):
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['point_clouds'].shape)).float()
        input_points_with_noise = batch_samples['point_clouds'] + xyz_noise*0.5
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device),
            'z_label1': batch_samples['z_label1'].to(device),
            'x_label1': batch_samples['x_label1'].to(device),
            'x_label2': batch_samples['x_label2'].to(device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(device),
            'labels': labels
        }
        with torch.no_grad():
            pred_results, losses = net(inputs)
            loss_sum += losses['trans_head'].item()
            loss_sum += losses['rot_z'].item()
            loss_sum += losses['rot_x'].item()
            log_state_dict = {
                    'eval trans1': losses['trans_head'].item(),
                    'eval vs': losses['vis_head'].item(),
                    'eval z': losses['rot_z'].item(),
                    'eval x': losses['rot_x'].item(),
                    }
            logger.update_state_dict(log_state_dict)                
    if gpus_is:
        if dist.get_rank() == 0:
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)
    else:
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)          
    return loss_sum



def train(start_epoch):
    global train_dataset
    min_loss = 1e10
    for epoch in range(start_epoch, MAX_EPOCH): 

        if epoch%TRAIN_DATA_HOLD_EPOCH == 0 or train_dataset is None:
            cid = int(epoch/TRAIN_DATA_HOLD_EPOCH) % len(TRAIN_CYCLE_RANGES)
            if gpus_is:
                if dist.get_rank() == 0:
                    print('Loading train dataset...')
                train_dataset = Parametric_Dataset(DATASET_DIR, DATA_OBJ_DIR,DATA_OBJ_DIR_TEMPLATE,DATA_OBJ_NAME,TRAIN_CYCLE_RANGES[cid], TRAIN_OBJ_RANGE, TRAIN_SCENE_RANGE, transforms=transforms)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=train_sampler)
                if dist.get_rank() == 0:
                    print('Test dataset loaded, test point cloud size:', len(train_dataset.dataset_dir))

            else:
                print('Loading train dataset...')
                train_dataset = Parametric_Dataset(DATASET_DIR, DATA_OBJ_DIR,DATA_OBJ_DIR_TEMPLATE,DATA_OBJ_NAME,TRAIN_CYCLE_RANGES[cid], TRAIN_OBJ_RANGE, TRAIN_SCENE_RANGE, transforms=transforms)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                print('Test dataset loaded, test point cloud size:', len(train_dataset.dataset_dir))

        bnm_scheduler.step(epoch) 
        lr_scheduler.step(epoch)

        if gpus_is:
            if dist.get_rank() == 0:
                logger.log_string('************** EPOCH %03d **************' % (epoch))
                logger.log_string(str(datetime.now()))
                logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
                logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))
        else:
            logger.log_string('************** EPOCH %03d **************' % (epoch))
            logger.log_string(str(datetime.now()))
            logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
            logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))

        train_one_epoch(train_loader,epoch)

        if epoch%EVAL_STAP == 0 and epoch>50:
            loss = eval_one_epoch(test_loader,epoch)
            if loss < min_loss:
                min_loss = loss
                save_checkpoint(os.path.join(log_dir, 'checkpoint.tar'), epoch, net, optimizer, loss)
                logger.log_string("Model saved in file: %s" % os.path.join(log_dir, 'checkpoint.tar'))
            
        if epoch%SAVE_STAP == 0 and epoch>50:
            save_checkpoint(os.path.join(log_dir, str(epoch)+'_'+'checkpoint.tar'), epoch, net, optimizer, loss)
    print(f'The training of{DATA_OBJ_NAME} is finished!')


# import models.utils.show3d_balls as show3d_balls
# def show_points(point_array, color_array=None, radius=3):
#     assert isinstance(point_array, list)
#     all_color = None
#     if color_array is not None:
#         assert len(point_array) == len(color_array)
#         # all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
#         # for i, c in enumerate(color_array):
#         #     all_color[i][:] = [[c[1],c[0],c[2]]]
#         all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
#         for i, c in enumerate(color_array):
#             all_color[i][:] = c
#         # all_color = np.array([np.zeros([point_array[0].shape[0], 3])])
#         # all_color[0][:] = color_array[0]
#         # all_color = all_color.reshape(-1, 3)
#         # color_array[1] = np.array(color_array[1])
#         # all_color = np.concatenate([all_color,color_array[1]], axis=0)
#         all_color = np.concatenate(all_color, axis=0)
#     all_points = np.concatenate(point_array, axis=0)
#     show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)


if __name__ == '__main__':
    train(start_epoch)
