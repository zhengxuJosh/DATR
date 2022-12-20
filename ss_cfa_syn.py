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

from models.segformer.segformer import Seg

from metrics.compute_iou import fast_hist, per_class_iu
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch
from models.segformer.segformer import Seg
from torch.utils import data

from dataset.adaption.sp13_dataset import synpass13DataSet
from dataset.adaption.dp13_dataset import densepass13TestDataSet
from dataset.adaption.dp13_dataset_ss import densepass13DataSet

import tqdm

#NAME_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole","light","sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motocycle","bicycle"]
NAME_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car']

def IFV(feat_1, feat_2, target0, target1, cen_bank_1, cen_bank_2, epoch):
    #feat_T.detach()
    size_f = (feat_1.shape[2], feat_1.shape[3])
    tar_feat_0 = nn.Upsample(size_f, mode='nearest')(target1.unsqueeze(1).float()).expand(feat_1.size())
    tar_feat_1 = nn.Upsample(size_f, mode='nearest')(target0.unsqueeze(1).float()).expand(feat_2.size())
    center_feat_S = feat_1.clone()
    center_feat_T = feat_2.clone()
    for i in range(19):
        mask_feat_0 = (tar_feat_0 == i).float()
        mask_feat_1 = (tar_feat_1 == i).float()
        center_feat_S = (1 - mask_feat_0) * center_feat_S + mask_feat_0 * ((mask_feat_0 * feat_1).sum(-1).sum(-1) / (mask_feat_0.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
        center_feat_T = (1 - mask_feat_1) * center_feat_T + mask_feat_1 * ((mask_feat_1 * feat_2).sum(-1).sum(-1) / (mask_feat_1.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

    center_feat_S = ((1 - 1 / (epoch + 1)) * cen_bank_1 + center_feat_S * 1 / (epoch + 1)) * 0.5
    center_feat_T = ((1 - 1 / (epoch + 1)) * cen_bank_2 + center_feat_T * 1 / (epoch + 1)) * 0.5
    # cosinesimilarity along C
    cos = nn.CosineSimilarity(dim=1)
    pcsim_feat_S = cos(feat_1, center_feat_S)
    pcsim_feat_T = cos(feat_2, center_feat_T)

    # FA
    mse = nn.MSELoss()
    loss = mse(pcsim_feat_S, pcsim_feat_T)
    # fa sfmx
    # kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=True)
    # loss = kl_loss(F.log_softmax(pcsim_feat_T), F.softmax(pcsim_feat_S).detach())
    # center sfmx
    # kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=True)
    # loss = kl_loss(F.log_softmax(center_feat_S), F.softmax(center_feat_T).detach())
    # center 
    # kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=True)
    # loss = kl_loss(pcsim_feat_S, pcsim_feat_T.detach())
    return loss, center_feat_S, center_feat_T

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    parser = argparse.ArgumentParser(description='pytorch implemention')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 6)')
    parser.add_argument('--iterations', type=int, default=30000, metavar='N',
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
    best_performance_dp, best_performance_sp = 0.0, 0.0
    
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
        syn_h, syn_w = 2048, 400#1024,512
        root_syn = '/hy-tmp/workplace/xuzheng/data/SynPASS'
        list_path = '/hy-tmp/workplace/xuzheng/CVPR/dataset/adaption/synpass_list/train.txt'
        syn_dataset = synpass13DataSet(root_syn, list_path, crop_size=(syn_h, syn_w), set='train')
        syn_train_sampler = DistributedSampler(syn_dataset, num_replicas=dist.get_world_size(), drop_last=True)
        syn_train_loader = torch.utils.data.DataLoader(syn_dataset,batch_size=args.batch_size,sampler=syn_train_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        
        val_list = '/hy-tmp/workplace/xuzheng/CVPR/dataset/adaption/synpass_list/val.txt'
        syn_val = synpass13DataSet(root_syn, val_list, crop_size=(1024,512))
        syn_val_loader = torch.utils.data.DataLoader(syn_val,batch_size=1,num_workers=1,pin_memory=True)
        
        test_list = '/hy-tmp/workplace/xuzheng/CVPR/dataset/adaption/synpass_list/test.txt'
        syn_test = synpass13DataSet(root_syn, test_list, crop_size=(syn_h, syn_w))
        syn_test_loader = torch.utils.data.DataLoader(syn_test,batch_size=1,num_workers=1,pin_memory=True)

        # DensePASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        root_dp = '/hy-tmp/workplace/xuzheng/data/DensePASS'
        list_path = '/hy-tmp/workplace/xuzheng/CVPR/dataset/adaption/densepass_list/val.txt'
        train_root = '/hy-tmp/workplace/xuzheng/data'
        train_list = '/hy-tmp/workplace/xuzheng/CVPR/dataset/adaption/densepass_list/train.txt'
        pass_train = densepass13DataSet(train_root, train_list, crop_size=(2048,400))
        pass_train_sampler = DistributedSampler(pass_train, num_replicas=dist.get_world_size(), drop_last=True)
        pass_train_loader = torch.utils.data.DataLoader(pass_train,batch_size=args.batch_size,sampler=pass_train_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        
        pass_dataset = densepass13TestDataSet(root_dp, list_path, crop_size=(2048,400),set='val')
        testloader = torch.utils.data.DataLoader(pass_dataset, batch_size=1, shuffle=True, num_workers=1,
                                        pin_memory=True)
        num_classes = 13
        NUM_CLASSES = 13
        w, h = 2048, 400
        # Models
        # ------------------------------------------------------------------------------------------------------------#
        model1 = Seg(backbone=args.backbone,num_classes=num_classes,embedding_dim=512,pretrained=True)
        model_path = "/hy-tmp/workplace/xuzheng/CVPR/exp/1023_[sup_ss]_[Seg_mit_nat_b2]_[pass_like_bs16_50epoch]/best_densepass.pth"
        model1.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")),strict=False)
        print('Model is', args.backbone)
        print('Load Model from', model_path)
        
        model1 = nn.SyncBatchNorm.convert_sync_batchnorm(model1)
        model1 = model1.to(args.local_rank)
        model1 = DistributedDataParallel(model1,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
        # Iterative dataloader
        # ------------------------------------------------------------------------------------------------------------#        
        syn_sup_loader = iter(syn_train_loader)
        pass_img_loader = iter(pass_train_loader)
        syn_length = len(syn_sup_loader)
        pass_length = len(pass_img_loader)

        if dist.get_rank() == 0:
            print(f'Panoramic Dataset length:{len(pass_train)};')
            print(f'SynPASS Dataset length:{len(syn_dataset)};')
            #syn_length = len(syn_dataset)
            #pass_length = len(train_DensePASS)
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        criterion_sup = nn.CrossEntropyLoss(reduction='mean', ignore_index=255) 
        optimizer1 = optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=0.0001)
        #optimizer2 = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=0.0001)
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        it = 1
        epoch = 1
        cen_bank_1, cen_bank_2 = torch.zeros(1, 512, 13, 64).to(args.local_rank), torch.zeros(1, 512, 13, 64).to(args.local_rank) 
        # Training Iterations
        # ------------------------------------------------------------------------------------------------------------# 
        for it in range(1, args.iterations + 1):
            if it % syn_length == 0:
                syn_train_loader.sampler.set_epoch(epoch)
                syn_sup_loader = iter(syn_train_loader)
            if it % pass_length == 0:
                pass_train_loader.sampler.set_epoch(epoch)
                pass_img_loader = iter(pass_train_loader)
            s_img, s_gt, _, _ = syn_sup_loader.__next__()#[1,400, 2048]
            s_img, s_gt = s_img.to(args.local_rank), s_gt.to(args.local_rank)
            p_img, p_gt, _, _ = pass_img_loader.__next__()#[1,3, 400, 2048]
            p_img, p_gt = p_img.to(args.local_rank), p_gt.to(args.local_rank) 
            # Model1 Prediction
            # ------------------------------------------------------------------------------------------------------------#        
            #input1 = torch.cat((s_img.permute(0,1,3,2), p_img),dim=0) #[2,3,400,2048]
            syn_pred, syn_feat = model1(s_img)
            pass_pred, pass_feat = model1(p_img)
            # Loss calculation
            # ------------------------------------------------------------------------------------------------------------#    
            # Supervised Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_sup_1 = criterion_sup(syn_pred,s_gt)
            writer.add_scalar('Model1 Sup Loss',loss_sup_1,it)
            # Pseudo Supervised Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_sup_2 = criterion_sup(pass_pred,p_gt)
            writer.add_scalar('Model1 Pseudo Sup Loss',loss_sup_2,it)
            # Feature Aggregation Loss
            # ------------------------------------------------------------------------------------------------------------#  
            city_target = s_gt
            erp_target = p_gt
            loss_fa_3, cen_1, cen_2 = IFV(pass_feat[3], syn_feat[3], erp_target, city_target, cen_bank_1, cen_bank_2, epoch)
            cen_bank_1 = cen_1.detach().clone() 
            cen_bank_2 = cen_2.detach().clone()
            loss_fa = loss_fa_3 #* args.alpha
            writer.add_scalar('Model1 FA Loss',loss_fa,it)
            # Model Total Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_1 = loss_sup_1 + loss_sup_2 + loss_fa_3
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
                lr_ = adjust_learning_rate_poly(optimizer1,it - 1500,it ,args.lr,1)
            # Print Loss
            # ------------------------------------------------------------------------------------------------------------#        
            if it % pass_length == 0 or it == 1:
                if dist.get_rank() == 0:
                    print(f'iter:{it};Model1 Total loss: {loss_1:.4f}')
                    print(f'iter:{it};Model1 Sup loss: {loss_sup_1:.4f}')
                    print(f'iter:{it};Model1 Pseudo Sup Loss: {loss_sup_2:.4f}')
                # Validation
                # ------------------------------------------------------------------------------------------------------------#    
                with torch.no_grad():    
                    if dist.get_rank() == 0:
                        model1.eval()
                        print(f'[Validation it: {it}] lr: {lr_}')
                        # DensePASS Validation
                        # ----------------------------------------------------------------------------------------------------# 
                        best_performance_dp = validation(num_classes, NUM_CLASSES, NAME_CLASSES, args.local_rank, testloader, model1, best_performance_dp, save_path, epoch, 'densepass')
                        epoch += 1
                        # SynPASS Validation
                        # ----------------------------------------------------------------------------------------------------# 
                        # best_performance_sp = validation(num_classes, NUM_CLASSES, NAME_CLASSES, args.local_rank, syn_val_loader, model1, best_performance_sp, save_path, epoch, 'synpass')
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
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))
    if name == 'densepass':
        if bestIoU > best_performance:
            best_performance = bestIoU
        torch.save(model1.module.state_dict(),save_path+"/best_densepass.pth")
        print('epoch:',epoch,name,'val_mIoU',bestIoU, 'best_model:', best_performance)
        writer.add_scalar('[DensePASS] val_mIOU:',bestIoU,epoch)
    if name == 'synpass':
        if bestIoU > best_performance:
            best_performance = bestIoU
        torch.save(model1.module.state_dict(),save_path+"/best_synpass.pth")
        print('epoch:',epoch,name,'val_mIoU',bestIoU, 'best_model:', best_performance)
        writer.add_scalar('[SynPASS] val_mIOU:',bestIoU,epoch)
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
     
    
