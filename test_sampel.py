import time

import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader as dataLoader1
from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate
from utils.dataloader import TrajectoryDataset
from model.STTODE import STTODENet
from sampler import Sampler
from utils.metrics import compute_ADE, compute_FDE, count_miss_samples
from utils.sddloader import SDD_Dataset
from show import *
sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter
from test import *

import urllib.request
import urllib.error

import requests
parser = argparse.ArgumentParser()


parser.add_argument('--frame', type=int, default=5000)
parser.add_argument('--sample_num', type=int, default=20)
#
# parser.add_argument('--sampler_epoch', type=int, default=200)
# parser.add_argument('--vae_epoch', type=int, default=80)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-3)
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

# vae_epoch
parser.add_argument('--vae_epoch', type=int, default=100)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
# parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

# training options
# parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default='step')
# model architecture
parser.add_argument('--pos_concat', type=bool, default=True)
parser.add_argument('--cross_motion_only', type=bool, default=True)

parser.add_argument('--tf_model_dim', type=int, default=256)
parser.add_argument('--tf_ff_dim', type=int, default=512)
parser.add_argument('--tf_nhead', type=int, default=8)
parser.add_argument('--tf_dropout', type=float, default=0.1)

parser.add_argument('--he_tf_layer', type=int, default=2)  # he = history encoder
parser.add_argument('--fe_tf_layer', type=int, default=2)  # fe = future encoder
parser.add_argument('--fd_tf_layer', type=int, default=2)  # fd = future decoder

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=3)
parser.add_argument('--asconv_layer_num', type=int, default=3)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--printfigure', type=bool, default=False)

parser.add_argument('--printone', type=bool, default=False)
# True
# loss config
parser.add_argument('--kld_weight', type=float, default=0.1)
parser.add_argument('--kld_min_clamp', type=float, default=10)
parser.add_argument('--recon_weight', type=float, default=5.0)
parser.add_argument('--lr_fix_epochs', type=int, default=10)

parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=100)

def test(STTODENet, sampler, loader_test, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()

    # total_cnt = 0
    # miss_cnt = 0

    for cnt, batch in enumerate(loader_test):
        seq_name = batch.pop()[0]
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame_idx))
        sys.stdout.flush()

        with torch.no_grad():
            STTODENet.set_data(batch, obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)
            dec_motion, _, _, attn_weights = sampler.forward(STTODENet)  # [N sn T 2]  # testing function
        # dec_motion = dec_motion.unsqueeze(1).repeat_interleave(args.sample_num, dim=1)
        dec_motion = dec_motion * traj_scale
        traj_gt = pred_traj_gt.transpose(1, 2) * traj_scale  # [N 2 T] -> [N T 2]

        # rearrange dec_motion
        agent_traj = []
        sample_motion = dec_motion.detach().cpu().numpy()  # [7 20 12 2]
        for i in range(sample_motion.shape[0]):  # traverse each person  list -> ped dimension
            agent_traj.append(sample_motion[i, :, :, :])
        traj_gt = traj_gt.detach().cpu().numpy()

        # calculate ade and fde and get the min value for 20 samples
        ade = compute_ADE(agent_traj, traj_gt)

        # #################################
        # import matplotlib.pyplot as plt
        #
        # # 示例数据
        # n=traj_gt.shape[0]
        # X_obs = []
        # Y_obs = []
        # obs_traj = obs_traj.detach().cpu().numpy()
        # for i in range(n):
        #     x_obs= obs_traj[i][0,:]
        #     y_obs = obs_traj[i][1,:]
        #     X_obs.append(x_obs)
        #     Y_obs.append(y_obs)
        # X = []
        # Y = []
        # for i in range(n):
        #     for j in range(20):
        #         x = agent_traj[i][j,:,0]
        #         y = agent_traj[i][j,:,1]
        #         X.append(x)
        #         Y.append(y)
        # X_GT = []
        # Y_GT = []
        # for i in range(n):
        #     x_GT = traj_gt[i][:,0]
        #     y_GT = traj_gt[i][:,1]
        #     X_GT.append(x_GT)
        #     Y_GT.append(y_GT)
        # # 创建一个折线图
        # plt.figure(figsize=(10, 5))
        # plt.scatter(X, Y, color='blue', marker='o')
        # plt.scatter(X_GT, Y_GT, color='red', marker='o')
        # plt.scatter(X_obs, Y_obs, color='green', marker='o')
        # title = str(frame_idx)
        # plt.title(f'Frame:{frame_idx}')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid(True)
        #
        #
        # #保存
        # # 动态生成文件名
        # filename = f'point_{frame_idx}.png'
        # save_path = 'F:\\gsa_plot_figure'
        # # 保存图表为PNG文件
        # save_path =  save_path+'\\'+args.dataset+'\\'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # save_file  = save_path+ filename
        # if not os.path.exists(save_file):
        #     plt.savefig(save_file)
        # plt.show()
        #
        #
        #
        #
        #
        # ###################################
        ade_meter.update(ade, n=STTODENet.agent_num)

        fde = compute_FDE(agent_traj, traj_gt)
        fde_meter.update(fde, n=STTODENet.agent_num)

        # miss_sample_num = count_miss_samples(agent_traj, traj_gt)
        # total_cnt += sample_motion.shape[0]
        # miss_cnt += miss_sample_num
        ################################
        # if args.printfigure:  # 显示图片
        #     # F:\\ref_figure
        #     # 读取背景图片
        #     if args.dataset == 'hotel':
        #         background_image_path = 'F:\\ref_figure\\seq_hotel\\'
        #     if args.dataset == 'eth':
        #         background_image_path = 'F:\\ref_figure\\seq_eth\\'
        #     if args.dataset == 'zara1':
        #         background_image_path = 'F:\\ref_figure\\crowds_zara01\\'
        #     if args.dataset == 'univ':
        #         background_image_path = 'F:\\ref_figure\\students003\\'
        #
        #     background_image = background_image_path + str((frame_idx)) + '.jpg'
        #
        #     flag = os.path.exists(background_image)
        #
        #     # 当前帧如果没有，即在附近寻找可用的帧
        #     if not (flag):
        #         for i in range(20):
        #             flag = os.path.exists(background_image_path + str((frame_idx - i)) + '.jpg')
        #             if (flag):
        #                 background_image = background_image_path + str((frame_idx - i)) + '.jpg'
        #                 break
        #
        #     bg_img = mpimg.imread(background_image)
        #     # angle = 0  # 旋转角度
        #     # bg_img = rotate(bg_img, angle, reshape=True)
        #     frame_one = args.frame
        #     if args.printone:
        #         if frame_idx == frame_one:
        #             if args.dataset == 'hotel':
        #                 fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #             if args.dataset == 'eth':
        #                 fig, ax = plt.subplots(figsize=(6.40, 4.80))
        #             if args.dataset == 'zara1':
        #                 fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #             if args.dataset == 'univ':
        #                 fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #     else:
        #         if args.dataset == 'hotel':
        #             # angle = 0  # 旋转角度顺时针
        #             # bg_img = rotate(bg_img, angle, reshape=True)
        #             fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #         if args.dataset == 'eth':
        #             fig, ax = plt.subplots(figsize=(6.40, 4.80))
        #         if args.dataset == 'zara1':
        #             fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #         if args.dataset == 'univ':
        #             fig, ax = plt.subplots(figsize=(7.20, 5.76))
        #     # 示例数据
        #     n = traj_gt.shape[0]  # 当前人数
        #     # n=2
        #
        #     obs_traj = obs_traj.detach().cpu().numpy()
        #
        #     ######################################################打印单帧版本
        #     print_one = args.printone
        #     if print_one:
        #         if frame_idx == frame_one:
        #             # 为不同的线输入不同的颜色
        #             Rgb = [[0, 185 / 255, 0],
        #                    [85 / 255, 102 / 255, 0],
        #                    [156 / 255, 102 / 255, 31 / 255],
        #                    [255 / 255, 153 / 255, 18 / 255],
        #                    [128 / 255, 42 / 255, 42 / 255],
        #                    [0, 204 / 255, 204 / 255]
        #
        #                    ]
        #             for num in range(n):  # 对每个行人进行画线
        #                 X_obs = []
        #                 Y_obs = []
        #                 for i in range(num, num + 1):
        #                     x_obs = obs_traj[i][0, :]
        #                     y_obs = obs_traj[i][1, :]
        #                     # x_obs,y_obs = mat_trans_realworld(x_obs,y_obs)
        #                     X_obs.append(x_obs)
        #                     Y_obs.append(y_obs)
        #                 X, Y = [], []
        #
        #                 for i in range(num, num + 1):
        #                     X_i = [0, 0, 0,
        #                            0, 0, 0,
        #                            0, 0, 0,
        #                            0, 0, 0]
        #                     Y_i = [0, 0, 0,
        #                            0, 0, 0,
        #                            0, 0, 0,
        #                            0, 0, 0]
        #                     for j in range(20):
        #                         x = agent_traj[i][j, :, 0]
        #                         y = agent_traj[i][j, :, 1]
        #                         ##########################画多条曲线
        #                         # X.append(x)
        #                         # Y.append(y)
        #                         #########################画单条曲线
        #                         X_i += x
        #                         Y_i += y
        #
        #                     X_i /= 20
        #                     Y_i /= 20
        #                     X.append(X_i)
        #                     Y.append(Y_i)
        #                 ###########################################
        #                 X_GT = []
        #                 Y_GT = []
        #
        #                 for i in range(num, num + 1):
        #                     x_GT = traj_gt[i][:, 0]
        #                     y_GT = traj_gt[i][:, 1]
        #                     # x_GT, y_GT = mat_trans_realworld(x_GT, y_GT)
        #                     X_GT.append(x_GT)
        #                     Y_GT.append(y_GT)
        #                 rgb_color1 = Rgb[num]
        #                 # 不同的行人用不同的颜色区分
        #                 # ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
        #                 ax.scatter(Y_GT, X_GT, color=rgb_color1, marker='o', s=30, label='gt')
        #                 ax.scatter(Y_obs, X_obs, color=rgb_color1, marker='o', s=30, label='obs')
        #
        #             ax.set_title(f'Frame:{frame_idx}')
        #             # ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
        #
        #             ##设置背景为白色
        #             height, width = bg_img.shape[0], bg_img.shape[1]
        #             bg_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # RGB值为255表示白色
        #             ax.imshow(bg_img, extent=[0, 12, 12.5, -3], aspect='auto')
        #
        #             # ################################ 创建一个散点图
        #             # rgb_color = (220 / 255, 207 / 255, 46 / 255)  # 转换后的RGB颜色值
        #             # if args.dataset == 'hotel':
        #             #     ax.scatter(Y, X, color=rgb_color, marker='o', s=5)
        #             #     ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10)
        #             #     ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10)
        #             #     ax.set_title(f'Frame:{frame_idx}')
        #             #     ax.imshow(bg_img, extent=[-10, 4, 5.8, -7], aspect='auto')
        #             # if args.dataset == 'eth':
        #             #     ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
        #             #     ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10, label='gt')
        #             #     ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10, label='obs')
        #             #     ax.set_title(f'Frame:{frame_idx}')
        #             #     ax.imshow(bg_img, extent=[-9, 20, 12.5, -3], aspect='auto')
        #             # if args.dataset == 'zara1':
        #             #     ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
        #             #     ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
        #             #     ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
        #             #     ax.set_title(f'Frame:{frame_idx}')
        #             #     ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
        #             # if args.dataset == 'univ':
        #             #     # ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
        #             #     # ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
        #             #     # ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
        #             #     ax.set_title(f'Frame:{frame_idx}')
        #             #     ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
        #
        #             ax.set_title(f'Frame:{frame_idx}-----ADE:{ade},FDE:{fde}')
        #             # 隐藏刻度
        #             ax.set_xticks([])
        #             ax.set_yticks([])
        #
        #             # 保存
        #             # 动态生成文件名
        #
        #             filename = f'point_{frame_idx}.png'
        #             save_path = 'F:\\sttode_groupnet3_plot_figure'
        #             # 保存图表为PNG文件
        #             save_path = save_path + '\\' + args.dataset + '\\'
        #             if not os.path.exists(save_path):
        #                 os.mkdir(save_path)
        #             save_file = save_path + filename
        #             if not os.path.exists(save_file):
        #                 plt.savefig(save_file)
        #             plt.show()
        #
        #     else:  ###绘制多帧，并且会存到本地文件夹
        #         #################################################################正常版本
        #         X_obs = []
        #         Y_obs = []
        #         for i in range(n):
        #             x_obs = obs_traj[i][0, :]
        #             y_obs = obs_traj[i][1, :]
        #             # x_obs,y_obs = mat_trans_realworld(x_obs,y_obs)
        #             X_obs.append(x_obs)
        #             Y_obs.append(y_obs)
        #         X, Y = [], []
        #
        #         for i in range(n):
        #             X_i = [0, 0, 0,
        #                    0, 0, 0,
        #                    0, 0, 0,
        #                    0, 0, 0]
        #             Y_i = [0, 0, 0,
        #                    0, 0, 0,
        #                    0, 0, 0,
        #                    0, 0, 0]
        #             for j in range(20):
        #                 x = agent_traj[i][j, :, 0]
        #                 y = agent_traj[i][j, :, 1]
        #                 ##########################画多条曲线，预测曲线绘制多条
        #                 X.append(x)
        #                 Y.append(y)
        #
        #                 #########################画单条曲线预测曲线
        #                 X_i += x
        #                 Y_i += y
        #
        #             X_i /= 20
        #             Y_i /= 20
        #             X.append(X_i)
        #             Y.append(Y_i)
        #             ###########################################
        #
        #         X_GT = []
        #         Y_GT = []
        #
        #         for i in range(n):
        #             x_GT = traj_gt[i][:, 0]
        #             y_GT = traj_gt[i][:, 1]
        #             # x_GT, y_GT = mat_trans_realworld(x_GT, y_GT)
        #             X_GT.append(x_GT)
        #             Y_GT.append(y_GT)
        #         #################################################################正常版本
        #
        #         ################################ 创建一个散点图
        #         rgb_color = (220 / 255, 207 / 255, 46 / 255)  # 转换后的RGB颜色值
        #         if args.dataset == 'hotel':
        #             ax.scatter(Y, X, color=rgb_color, marker='o', s=5)
        #             ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10)
        #             ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10)
        #
        #             ax.imshow(bg_img, extent=[-10, 5, 5.8, -7],
        #                       aspect='auto')  # ax.imshow(bg_img,extent=[-10, 4, 5.8, -7], aspect='auto')右边太往右了
        #         if args.dataset == 'eth':
        #             ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
        #             ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10, label='gt')
        #             ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10, label='obs')
        #
        #             ax.imshow(bg_img, extent=[-9, 20, 12.5, -3], aspect='auto')
        #         if args.dataset == 'zara1':
        #             ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
        #             ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
        #             ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
        #
        #             ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
        #         if args.dataset == 'univ':
        #             ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
        #             ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
        #             ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
        #
        #             ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
        #         ax.set_title(f'Frame:{frame_idx}-----ADE:{ade},FDE:{fde}')
        #         # 隐藏刻度
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #
        #         # 保存
        #         # 动态生成文件名
        #
        #         filename = f'point_{frame_idx}.png'
        #         save_path = 'F:\\sttode_plot_figure'
        #         # 保存图表为PNG文件
        #         save_path = save_path + '\\' + args.dataset + '\\'
        #         if not os.path.exists(save_path):
        #             os.mkdir(save_path)
        #         save_file = save_path + filename
        #         if not os.path.exists(save_file):
        #             plt.savefig(save_file)
        #         plt.show()
        #
        #         ###################################

        if args.dataset == 'sdd':
            try:
                print(f'Frame={frame_idx},ade={ade},fde={fde}')
                show_sdd(args, frame_idx, traj_gt, obs_traj*50, agent_traj, ade, fde)
            except requests.exceptions.HTTPError as e:#urllib.error.HTTPError as e:
                if e.response.status_code == 500:
                    a=1
                else:
                    a=2
            except urllib.error.URLError as e:
                b=1
            except Exception as e:
                c=1
        else:
            show_eth(args, frame_idx, traj_gt, obs_traj, agent_traj, ade, fde)
            # show_eth(args, 5000, pred_traj_gt_world.detach().cpu().numpy(), obs_world, pred_traj, ade, fde)
        # show_eth(args,5000,traj_gt ,obs_traj_world.permute(0,2,1) ,pred_traj, ade,fde)
        '''
        traj_gt [N 12 2]
        obs_traj [N 2 8]
        pred_traj LIST[N] [20 12 2]
        '''
        # miss_sample_num = count_miss_samples(agent_traj, traj_gt)
        # total_cnt += sample_motion.shape[0]
        # miss_cnt += miss_sample_num

    # miss_rate = float(miss_cnt) / float(total_cnt)

    # miss_rate = float(miss_cnt) / float(total_cnt)

    return ade_meter, fde_meter  # , miss_rate


def main(args):
    args = parser.parse_args()
    if args.dataset == 'nba':
        pass
    else:
        args.past_length = 8
        args.future_length = 12
    data_set = './datasets/' + args.dataset + '/'

    prepare_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    ###################################
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    traj_scale = 1.0
    if args.dataset == 'nba':
        test_dset = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            training=False)

        test_loader = DataLoader(
            test_dset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            collate_fn=seq_collate,
            pin_memory=True)
    else:
        data_set = './datasets/' + args.dataset + '/'

        traj_scale = 1.0
        if args.dataset == 'eth':
            args.max_train_agent = 32
            # traj_scale = 2.0
            # args.fe_out_mlp_dim = [512, 256]
            # args.fd_out_mlp_dim = [512, 256]
            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=args.past_length,
                pred_len=args.future_length,
                skip=1, traj_scale=traj_scale)

        elif args.dataset == 'sdd':
            traj_scale = args.sdd_scale
            dset_test = SDD_Dataset(
                data_set + 'test/',
                obs_len=args.past_length,
                pred_len=args.future_length,
                skip=1, traj_scale=args.sdd_scale)
        else:
            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=args.past_length,
                pred_len=args.future_length,
                skip=1, traj_scale=traj_scale)

        test_loader = dataLoader1(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=0)

    net = STTODENet(args,device)
    sampler = Sampler(args)
    # load STTODENet model
    vae_dir = './saved_models/' + args.dataset + '/'
    all_vae_models = os.listdir(vae_dir)
    if len(all_vae_models) == 0:
        print('VAE model not found!')
        return

    sampler_dir = './checkpoints/' + args.dataset + '/sampler/'
    all_sampler_models = os.listdir(sampler_dir)
    if len(all_sampler_models) == 0:
        print('sampler model not found!')
        return

    minade = 100000
    SAMPLER_EPOCH = 0
    VAE_EPOCH = 0
    minfde =0




    for y in all_sampler_models[:]:

        if args.dataset == 'hotel':
            y = 'model_%04d.p' % 2#  80:44  22 70: 42   60:40  #90:27
        elif args.dataset == 'eth':
            y = y#'model_%04d.p' % 10  #vae 80  22  28  14  #100vae 24,26,28  # 80vae :31 #70:10
        elif args.dataset == 'zara1':
            y = 'model_%04d.p' % 1 # 46# 58 36   50 # 46 56 36 50 #120:36  40  #80 :29   40 38
        elif args.dataset == 'zara2':
            y = y#'model_%04d.p' %28  # 42  48 32  # 60  36  # 60vae 38 36
        elif args.dataset == 'univ':
            y = y#'model_%04d.p' % 24  # zara 60  #hotel 75 45  #135 :32
        else:
            y = y#'model_%04d.p' % 102 #y 100
    # if default_sampler_model not in all_sampler_models:
    #     default_sampler_model = all_sampler_models[-1]
    # load sampler model
        sampler_path = os.path.join(sampler_dir, y)
        model_cp = torch.load(sampler_path, map_location='cpu')
        sampler.load_state_dict(model_cp)
        # torch.save(model_cp['model_dict'], cp_path)
        print('loading model from checkpoint: %s' % sampler_path)



        sampler.set_device(device)
        sampler.eval()
        for x in all_vae_models[-1:]:
            if args.dataset =='hotel':
                x = x#'model_%04d.p' % 90#    45  # 90#65     #   102  #90:101
            elif args.dataset == 'eth':
                x = x#'model_%04d.p' % 106 # 45 100 106
            elif args.dataset == 'zara1':
                x = x#'model_%04d.p' % 147 # zara 60  #hotel 75 55   #141    115   133
            elif args.dataset == 'zara2':
                x = x#'model_%04d.p' % 80  # zara 60  #hotel 75
            elif args.dataset == 'univ':
                x = x#'model_%04d.p' % 60 #72  60  45
            else:
                x = 'model_%04d.p' % 80  # sdd 100
        # if default_vae_model not in all_vae_models:
        #     x = all_vae_models[-1]
            vae_path = os.path.join(vae_dir, x)
            print('loading model from checkpoint: %s' % vae_path)
            # model_cp = torch.load(vae_path, map_location='cpu')
            # # torch.save(model_cp['model_dict'], cp_path)
            # net.load_state_dict(model_cp)
            model_load = torch.load(vae_path, map_location='cpu')
            net.load_state_dict(model_load['model_dict'])

            net.set_device(device)
            net.eval()
            if args.dataset == 'nba':
                test_model_all(net, test_loader, args)
            else:
                # run testing
                ade_meter, fde_meter = test(net, sampler, test_loader, traj_scale)

                print('-' * 20 + ' STATS ' + '-' * 20)
                print('ADE: %.4f' % ade_meter.avg)
                print('FDE: %.4f' % fde_meter.avg)
                #记录最小的ade以及fde

                if ade_meter.avg <minade:
                    minade = ade_meter.avg
                    minfde = fde_meter.avg
                    print('minADE: %.4f' % minade)
                    SAMPLER_EPOCH = sampler_path
                    VAE_EPOCH =  vae_path

                print('-' * 30 + ' STATS ' + '-' * 30)
                print('result:minADE: %.4f' % minade)
                print('result:minFDE: %.4f' % minfde)
                print('result:SAMPLER_EPOCH:' +str(SAMPLER_EPOCH))
                print('result:SAMPLER_EPOCH:' +str(VAE_EPOCH))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

