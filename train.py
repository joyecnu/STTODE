import os
import sys
import argparse

import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader as dataLoader1
from data.dataloader_nba import NBADataset, seq_collate
from model.STTODE import STTODENet
from utils.dataloader import TrajectoryDataset
from utils.sddloader import SDD_Dataset
from model_structure import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)
parser.add_argument('--iternum_print', type=int, default=100)

parser.add_argument('--ztype', default='gaussian')
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--hyper_scales', nargs='+', type=int,default=[5,11])
parser.add_argument('--num_decompose', type=int, default=2)
parser.add_argument('--min_clip', type=float, default=2.0)

parser.add_argument('--model_save_dir', default='saved_models/')
parser.add_argument('--model_save_epoch', type=int, default=5)

parser.add_argument('--epoch_continue', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)
parser.add_argument('--sdd_scale', type=float, default=50.0)


def train(model,optimizer,scheduler,train_loader,epoch,args):
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    if args.dataset == 'nba':
        for data in train_loader:
            model.set_data_nba(data)
            total_loss,loss_pred,loss_recover,loss_kl,loss_diverse = model.forward()
            """ optimize """
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iter_num % args.iternum_print == 0:
                print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
                .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
            iter_num += 1
    else:
        for cnt, batch in enumerate(train_loader):

            seq_name = batch.pop()[0]
            frame_idx = int(batch.pop()[0])
            batch = [tensor[0].cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
            non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

            model.set_data(batch, obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)  # [T N or N*sn 2]

            total_loss,loss_pred,loss_recover,loss_kl,loss_diverse = model.forward()
            """ optimize """
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iter_num % args.iternum_print == 0:
                print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
                .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
            iter_num += 1

    scheduler.step()
    model.step_annealer()




def main():
    args = parser.parse_args()
    if args.dataset == 'nba':
        pass
    else:
        args.past_length = 8
        args.future_length = 12

    """ setup """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print('device:',device)
    print(args)

    """ model & optimizer """
    model = STTODENet(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

    """ dataloader """
    data_set = './datasets/' + args.dataset + '/'

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    traj_scale = 1.0
    if args.dataset == 'eth':
        args.max_train_agent = 32
        # traj_scale = 2.0
        # args.fe_out_mlp_dim = [512, 256]
        # args.fd_out_mlp_dim = [512, 256]
        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=args.past_length,
            pred_len=args.future_length,
            skip=1, traj_scale=traj_scale)

    elif args.dataset == 'sdd':
        args.traj_scale = args.sdd_scale
        dset_train = SDD_Dataset(
            data_set + 'train/',
            obs_len=args.past_length,
            pred_len=args.future_length,
            skip=1, traj_scale=args.sdd_scale)
    elif args.dataset == 'nba':
        dset_train = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            training=True)


    else:
        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=args.past_length,
            pred_len=args.future_length,
            skip=1, traj_scale=traj_scale)
    if(1):
        if args.dataset == 'nba':
            train_loader = dataLoader1(
                dset_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=seq_collate,
                pin_memory=True)
        else:
            train_loader = dataLoader1(
                dset_train,
                batch_size=1,
                shuffle=True,
                num_workers=0)


    """ Loading if needed """
    if args.epoch_continue > 0:
        checkpoint_path = os.path.join(args.model_save_dir+args.dataset+'/', ('model_%04d.p')%args.epoch_continue)
        print('load model from: {checkpoint_path}')
        model_load = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(model_load['model_dict'])
        if 'optimizer' in model_load:
            optimizer.load_state_dict(model_load['optimizer'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()


        if 'scheduler' in model_load:
            scheduler.load_state_dict(model_load['scheduler'])

    """ start training """
    model.set_device(device)
    checkpoint_dir = args.model_save_dir + args.dataset
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_structure(model)
    for epoch in range(args.epoch_continue, args.num_epochs):
        train(model,optimizer,scheduler,train_loader, epoch,args)
        """ save model """
        if (epoch + 1) % args.model_save_epoch == 0:
            model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'model_cfg': args}
            saved_path = os.path.join(checkpoint_dir, 'model_%04d.p') % (epoch + 1)
            torch.save(model_saved, saved_path)





if __name__ == '__main__':
    main()


