import warnings 
warnings.filterwarnings('ignore')

import argparse
import random
import os
import time,datetime
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.compute_iou import fast_hist, per_class_iu
from dataset.densepass_dataset import densepassTestDataSet
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils import data

from dataset.adaption.sp13_dataset import synpass13DataSet
from dataset.adaption.sp22_dataset import synpassDataSet
from dataset.adaption.dp13_dataset import densepass13TestDataSet

import tqdm

from models.DATR.DATR import DATR

#NAME_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole","light","sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motocycle","bicycle"]
NAME_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car']


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    parser = argparse.ArgumentParser(description='pytorch implemention')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 6)')
    parser.add_argument('--num_epochs', type=int, default=30000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-5, metavar='LR',
                        help='learning rate (default: 6e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--save_root', default = '',
                        help='Please add your model save directory') 
    parser.add_argument('--exp_name', default = '',
                        help='')
    parser.add_argument('--backbone',  type=str, default = '',
                        help='')        
    parser.add_argument('--sup_set', type=str, default='train', help='supervised training set')
    parser.add_argument('--cutmix', default =False, help='cutmix')
    #================================hyper parameters================================#
    parser.add_argument('--alpha', type=float, default =0.5, help='alpha')
    parser.add_argument('--lamda', type=float, default =0.001, help='lamda')
    parser.add_argument('--dis_lr', type=float, default =0.001, help='dis_lr')
    #================================================================================#
    args = parser.parse_args()
    best_performance_dp, best_performance_sp_v, best_performance_sp_t = 0.0, 0.0, 0.0
    
    save_path = "{}{}".format(args.save_root,args.exp_name)
    cur_time=str(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"))
    writer = SummaryWriter(log_dir=save_path)

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)   

    torch.cuda.set_device(args.local_rank)
    with torch.cuda.device(args.local_rank):
        dist.init_process_group(backend='nccl',init_method='env://') #nccl
        if dist.get_rank() == 0:
            print(args)
            print('init cnn lr: {}, batch size: {}, gpus:{}'.format(args.lr, args.batch_size, dist.get_world_size()))

        # SynPASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        syn_h, syn_w = 1024,512
        root_syn = '/hpc/users/CONNECT/xuzheng/data/SynPASS'
        list_path = '/hpc/users/CONNECT/xuzheng/data/SynPASS/train.txt'
        syn_dataset = synpassDataSet(root_syn, list_path, crop_size=(syn_h, syn_w), set='train')
        syn_train_sampler = DistributedSampler(syn_dataset, num_replicas=dist.get_world_size(), drop_last=True)
        syn_train_loader = torch.utils.data.DataLoader(syn_dataset,batch_size=args.batch_size,sampler=syn_train_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        
        val_list = '/hpc/users/CONNECT/xuzheng/data/SynPASS/val.txt'
        syn_val = synpassDataSet(root_syn, val_list, crop_size=(2048,400))
        syn_val_loader = torch.utils.data.DataLoader(syn_val,batch_size=1,num_workers=1,pin_memory=True)
        
        test_list = '/hpc/users/CONNECT/xuzheng/data/SynPASS/test.txt'
        syn_test = synpassDataSet(root_syn, test_list, crop_size=(syn_h, syn_w))
        syn_test_loader = torch.utils.data.DataLoader(syn_test,batch_size=1,num_workers=1,pin_memory=True)

        # DensePASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        root_dp = '/hpc/users/CONNECT/xuzheng/omni_seg/DensePASS'
        list_path = '/hpc/users/CONNECT/xuzheng/Trans4PASS/adaptations/dataset/densepass_list/val.txt'
        pass_dataset = densepass13TestDataSet(root_dp, list_path, crop_size=(2048,400),set='val')
        testloader = torch.utils.data.DataLoader(pass_dataset, batch_size=1, shuffle=True, num_workers=1,
                                        pin_memory=True)
        num_classes = 22
        NUM_CLASSES = 22
        w, h = 2048, 400
        # Models
        # ------------------------------------------------------------------------------------------------------------#
        model1 = DATR(backbone=args.backbone,num_classes=num_classes,embedding_dim=512,pretrained=True)
        model1 = nn.SyncBatchNorm.convert_sync_batchnorm(model1)
        model1 = model1.to(args.local_rank)
        model1 = DistributedDataParallel(model1,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
        # # Iterative dataloader
        # ------------------------------------------------------------------------------------------------------------#        
        #syn_sup_loader = iter(syn_train_loader)
        #pass_img_loader = iter(pass_train_loader)
        if dist.get_rank() == 0:
        #print(f'Panoramic Dataset length:{len(train_DensePASS)};')
            print(f'SynPASS Dataset length:{len(syn_dataset)};')
        syn_length = len(syn_dataset)
            #pass_length = len(train_DensePASS)
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        criterion_sup = nn.CrossEntropyLoss(reduction='mean', ignore_index=255) 
        optimizer1 = optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=0.0001)
        #optimizer2 = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=0.0001)
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        it = 1
        iterations = args.num_epochs * syn_length / args.batch_size

        # Training Iterations
        # ------------------------------------------------------------------------------------------------------------# 
        for epoch in range(args.num_epochs):
            syn_train_loader.sampler.set_epoch(epoch)
            with tqdm.tqdm(syn_train_loader, ncols=100) as pbar:
                for s_img, s_gt, _, _ in pbar:
                    pbar.set_description('epoch: {}/{}'.format(epoch + 1, args.num_epochs))
                    s_img, s_gt = s_img.to(args.local_rank), s_gt.to(args.local_rank)
                    # Model1 Prediction
                    # ------------------------------------------------------------------------------------------------------------#
                    syn_pred, _ = model1(s_img)
                    # Loss calculation
                    # ------------------------------------------------------------------------------------------------------------#    
                    # Supervised Loss
                    # ------------------------------------------------------------------------------------------------------------#        
                    loss_sup_1 = criterion_sup(syn_pred,s_gt)
                    writer.add_scalar('Model1 Sup Loss',loss_sup_1,it)
                    # Model Total Loss
                    # ------------------------------------------------------------------------------------------------------------#        
                    loss_1 = loss_sup_1 
                    # Model Optimization
                    # ------------------------------------------------------------------------------------------------------------#        
                    optimizer1.zero_grad()
                    loss_1.backward()
                    optimizer1.step()
                    # ------------------------------------------------------------------------------------------------------------# 
                    it += 1 
                    base_lr = args.lr
                    if it <= 1500:
                        lr_ = base_lr * (it / 1500)
                        for param_group in optimizer1.param_groups:
                            param_group['lr'] = lr_ 
                    else:
                        lr_ = adjust_learning_rate_poly(optimizer1,it - 1500,iterations ,args.lr,1)
            # Print Loss
            # ------------------------------------------------------------------------------------------------------------#        
            if dist.get_rank() == 0:
                print(f'epoch:{epoch};Model1 Total loss: {loss_1:.4f}')
                print(f'epoch:{epoch};Model1 Sup loss: {loss_sup_1:.4f}')
            # Validation
            # ------------------------------------------------------------------------------------------------------------#    
            with torch.no_grad():    
                if dist.get_rank() == 0:
                    model1.eval()
                    print(f'[Validation it: {it}] lr: {lr_}')
                    # DensePASS Validation
                    # ----------------------------------------------------------------------------------------------------# 
                    # best_performance_dp = validation(num_classes, NUM_CLASSES, NAME_CLASSES, args.local_rank, testloader, model1, best_performance_dp, save_path, epoch, 'densepass')
                    # SynPASS Validation
                    # ----------------------------------------------------------------------------------------------------# 
                    best_performance_sp_v = validation(num_classes, NUM_CLASSES, NAME_CLASSES, args.local_rank, syn_val_loader, model1, best_performance_sp_v, save_path, epoch, 'synpass_val')
                    best_performance_sp_t = validation(num_classes, NUM_CLASSES, NAME_CLASSES, args.local_rank, syn_test_loader, model1, best_performance_sp_t, save_path, epoch, 'synpass_test')
                    model1.train()
    
def validation(num_classes, NUM_CLASSES, NAME_CLASSES, device, testloader, model1, best_performance, save_path, epoch, name):
    writer = SummaryWriter(log_dir=save_path)
    hist = np.zeros((num_classes, num_classes))
    for index, batch in enumerate(testloader):
        # if index % 100 == 0:
        #     print ('%d processd' % index)
        image, label, _, _ = batch
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            output, _ = model1(image)
            #_, output = model1(image)
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    # for ind_class in range(num_classes):
    #     print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))
    if name == 'densepass':
        if bestIoU > best_performance:
            best_performance = bestIoU
        torch.save(model1.module.state_dict(),save_path+"/best_densepass.pth")
        print('epoch:',epoch,name,'val_mIoU',bestIoU, 'best_model:', best_performance)
        writer.add_scalar('[DensePASS] val_mIOU:',bestIoU,epoch)
    if name == 'synpass_val':
        if bestIoU > best_performance:
            best_performance = bestIoU
        torch.save(model1.module.state_dict(),save_path+"/best_synpass_val.pth")
        print('[SynPASS] val_mIOU: epoch:',epoch,name,'val_mIoU',bestIoU, 'best_model:', best_performance)
        writer.add_scalar('[SynPASS] val_mIOU:',bestIoU,epoch)
    if name == 'synpass_test':
        if bestIoU > best_performance:
            best_performance = bestIoU
        torch.save(model1.module.state_dict(),save_path+"/best_synpass_test.pth")
        print('[SynPASS] test_mIOU: epoch:',epoch,name,'test_mIoU',bestIoU, 'best_model:', best_performance)
        writer.add_scalar('[SynPASS] test_mIOU:',bestIoU,epoch)
    return best_performance

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    print('file name: ', __file__)
    setup_seed(1234)
    main()
     
    
