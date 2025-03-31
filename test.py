import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import random
sys.path.append(os.getcwd())
import torch
from data.dataloader_nba import NBADataset, seq_collate
from model.STTODE import STTODENet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.lines as mlines
from utils.dataloader import TrajectoryDataset
from utils.sddloader import SDD_Dataset
from utils.utils import prepare_seed, AverageMeter
from utils.metrics import compute_ADE, compute_FDE, count_miss_samples
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model_names', default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_save_dir', default='saved_models/')
parser.add_argument('--vis', action='store_true', default=True)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)
parser.add_argument('--sdd_scale', type=float, default=50.0)#50.0)
parser.add_argument('--dataset', default='nba')

# 打印单帧
parser.add_argument('--printone', type=bool, default=False)
#是否显示图片
parser.add_argument('--printfigure', type=bool, default=False)
#单独显示的帧
parser.add_argument('--frame', type=int, default=900)

class Constant:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 100
    Y_MIN = 0
    Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
    MESSAGE = 'You can rerun the script and choose any event from 0 to '

def draw_result(args,future,past,mode='pre'):
    # b n t 2
    print('drawing...')
    trajs = np.concatenate((past,future), axis = 2)
    batch = trajs.shape[0]
    for idx in range(50):
        plt.clf()
        traj = trajs[idx]
        traj = traj*94/28
        actor_num = traj.shape[0]
        length = traj.shape[1]
        
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                        ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid

        colorteam1 = 'dodgerblue'
        colorteam2 = 'orangered'
        colorball = 'limegreen'
        colorteam1_pre = 'skyblue'
        colorteam2_pre = 'lightsalmon'
        colorball_pre = 'mediumspringgreen'
		
        for j in range(actor_num):
            if j < 5:
                color = colorteam1
                color_pre = colorteam1_pre
            elif j < 10:
                color = colorteam2
                color_pre = colorteam2_pre
            else:
                color_pre = colorball_pre
                color = colorball
            for i in range(length):
                points = [(traj[j,i,0],traj[j,i,1])]
                (x, y) = zip(*points)
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre,s=20,alpha=1)
                else:
                    plt.scatter(x, y, color=color,s=20,alpha=1)

            for i in range(length-1):
                points = [(traj[j,i,0],traj[j,i,1]),(traj[j,i+1,0],traj[j,i+1,1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre,alpha=0.5,linewidth=2)
                else:
                    plt.plot(x, y, color=color,alpha=1,linewidth=2)

        court = plt.imread("datasets/nba/court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN],alpha=0.5)
        vispath = 'vis/' + args.dataset +'/'
        if not os.path.exists(vispath):
            os.makedirs(vispath)
        if mode == 'pre':
            # plt.savefig('vis/nba/'+str(idx)+'pre.png')
            plt.savefig(vispath+str(idx)+'pre.png')
        else:
            # plt.savefig('vis/nba/'+str(idx)+'gt.png')
            plt.savefig(vispath+str(idx)+'gt.png')
    print('ok')
    return 


def vis_result(test_loader, args):
    total_num_pred = 0
    all_num = 0

    for data in test_loader:
        future_traj = np.array(data['future_traj']) * args.traj_scale # B,N,T,2
        with torch.no_grad():
            prediction = model.inference(data)
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu()) #(BN,20,T,2)
        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]

        y = np.reshape(future_traj,(batch*actor_num,args.future_length, 2))
        y = y[None].repeat(20,axis=0)
        error = np.mean(np.linalg.norm(y- prediction,axis=3),axis=2)
        indices = np.argmin(error, axis = 0)
        best_guess = prediction[indices,np.arange(batch*actor_num)]
        best_guess = np.reshape(best_guess, (batch,actor_num, args.future_length, 2))
        gt = np.reshape(future_traj,(batch,actor_num,args.future_length, 2))
        # previous_3D = np.reshape(previous_3D,(batch,actor_num,args.future_length, 2))
        previous_3D = np.reshape(gt,(batch,actor_num,args.future_length, 2))

        draw_result(args,best_guess,previous_3D)
        draw_result(args, gt,previous_3D,mode='gt')
    return 

@torch.no_grad()
def eval(args, loader_test, model, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()

    num_frame = 0
    inference_time = 0
    total_num_agents = 0
    for cnt, batch in enumerate(loader_test):
        num_frame += 1
        seq_name = batch.pop()[0]
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame_idx))
        sys.stdout.flush()
        with torch.no_grad():
            model.set_data(batch, obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)
            start_time = time.time()  # 推理开始时间
            dec_motion = model.inference(batch)
            # pred = pred.permute(1, 0, 2)  # [N,T,2]
            dec_motion = dec_motion.permute(1, 0, 2, 3)  # [N,sn,T,2]
            end_time = time.time()  # 获取结束时间
        inference_time += (end_time - start_time)
        total_num_agents += dec_motion.shape[0]  # 当前场景下的人数
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
        ade_meter.update(ade, n=model.agent_num)

        fde = compute_FDE(agent_traj, traj_gt)
        fde_meter.update(fde, n=model.agent_num)

        ################################
        if args.printfigure:  # 显示图片
            # F:\\ref_figure
            # 读取背景图片
            if args.dataset == 'hotel':
                background_image_path = 'F:\\ref_figure\\seq_hotel\\'
            if args.dataset == 'eth':
                background_image_path = 'F:\\ref_figure\\seq_eth\\'
            if args.dataset == 'zara1':
                background_image_path = 'F:\\ref_figure\\crowds_zara01\\'
            if args.dataset == 'univ':
                background_image_path = 'F:\\ref_figure\\students003\\'

            background_image = background_image_path + str((frame_idx)) + '.jpg'

            flag = os.path.exists(background_image)

            # 当前帧如果没有，即在附近寻找可用的帧
            if not (flag):
                for i in range(20):
                    flag = os.path.exists(background_image_path + str((frame_idx - i)) + '.jpg')
                    if (flag):
                        background_image = background_image_path + str((frame_idx - i)) + '.jpg'
                        break

            bg_img = mpimg.imread(background_image)
            # angle = 0  # 旋转角度
            # bg_img = rotate(bg_img, angle, reshape=True)
            frame_one = args.frame
            if args.printone:
                if frame_idx == frame_one:
                    if args.dataset == 'hotel':
                        fig, ax = plt.subplots(figsize=(7.20, 5.76))
                    if args.dataset == 'eth':
                        fig, ax = plt.subplots(figsize=(6.40, 4.80))
                    if args.dataset == 'zara1':
                        fig, ax = plt.subplots(figsize=(7.20, 5.76))
                    if args.dataset == 'univ':
                        fig, ax = plt.subplots(figsize=(7.20, 5.76))
            else:
                if args.dataset == 'hotel':
                    # angle = 0  # 旋转角度顺时针
                    # bg_img = rotate(bg_img, angle, reshape=True)
                    fig, ax = plt.subplots(figsize=(7.20, 5.76))
                if args.dataset == 'eth':
                    fig, ax = plt.subplots(figsize=(6.40, 4.80))
                if args.dataset == 'zara1':
                    fig, ax = plt.subplots(figsize=(7.20, 5.76))
                if args.dataset == 'univ':
                    fig, ax = plt.subplots(figsize=(7.20, 5.76))
            # 示例数据
            n = traj_gt.shape[0]  # 当前人数
            # n=2

            obs_traj = obs_traj.detach().cpu().numpy()

            ######################################################打印单帧版本
            print_one = args.printone
            if print_one:
                if frame_idx == frame_one:
                    # 为不同的线输入不同的颜色
                    Rgb = [[0, 185 / 255, 0],
                           [85 / 255, 102 / 255, 0],
                           [156 / 255, 102 / 255, 31 / 255],
                           [255 / 255, 153 / 255, 18 / 255],
                           [128 / 255, 42 / 255, 42 / 255],
                           [0, 204 / 255, 204 / 255]

                           ]
                    for num in range(n):  # 对每个行人进行画线
                        X_obs = []
                        Y_obs = []
                        for i in range(num, num + 1):
                            x_obs = obs_traj[i][0, :]
                            y_obs = obs_traj[i][1, :]
                            # x_obs,y_obs = mat_trans_realworld(x_obs,y_obs)
                            X_obs.append(x_obs)
                            Y_obs.append(y_obs)
                        X, Y = [], []

                        for i in range(num, num + 1):
                            X_i = [0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0]
                            Y_i = [0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0]
                            for j in range(20):
                                x = agent_traj[i][j, :, 0]
                                y = agent_traj[i][j, :, 1]
                                ##########################画多条曲线
                                # X.append(x)
                                # Y.append(y)
                                #########################画单条曲线
                                X_i += x
                                Y_i += y

                            X_i /= 20
                            Y_i /= 20
                            X.append(X_i)
                            Y.append(Y_i)
                        ###########################################
                        X_GT = []
                        Y_GT = []

                        for i in range(num, num + 1):
                            x_GT = traj_gt[i][:, 0]
                            y_GT = traj_gt[i][:, 1]
                            # x_GT, y_GT = mat_trans_realworld(x_GT, y_GT)
                            X_GT.append(x_GT)
                            Y_GT.append(y_GT)
                        rgb_color1 = Rgb[num]
                        # 不同的行人用不同的颜色区分
                        # ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
                        ax.scatter(Y_GT, X_GT, color=rgb_color1, marker='o', s=30, label='gt')
                        ax.scatter(Y_obs, X_obs, color=rgb_color1, marker='o', s=30, label='obs')

                    ax.set_title(f'Frame:{frame_idx}')
                    # ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')

                    ##设置背景为白色
                    height, width = bg_img.shape[0], bg_img.shape[1]
                    bg_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # RGB值为255表示白色
                    ax.imshow(bg_img, extent=[0, 12, 12.5, -3], aspect='auto')

                    # ################################ 创建一个散点图
                    # rgb_color = (220 / 255, 207 / 255, 46 / 255)  # 转换后的RGB颜色值
                    # if args.dataset == 'hotel':
                    #     ax.scatter(Y, X, color=rgb_color, marker='o', s=5)
                    #     ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10)
                    #     ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10)
                    #     ax.set_title(f'Frame:{frame_idx}')
                    #     ax.imshow(bg_img, extent=[-10, 4, 5.8, -7], aspect='auto')
                    # if args.dataset == 'eth':
                    #     ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
                    #     ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10, label='gt')
                    #     ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10, label='obs')
                    #     ax.set_title(f'Frame:{frame_idx}')
                    #     ax.imshow(bg_img, extent=[-9, 20, 12.5, -3], aspect='auto')
                    # if args.dataset == 'zara1':
                    #     ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                    #     ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                    #     ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
                    #     ax.set_title(f'Frame:{frame_idx}')
                    #     ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
                    # if args.dataset == 'univ':
                    #     # ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                    #     # ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                    #     # ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
                    #     ax.set_title(f'Frame:{frame_idx}')
                    #     ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')

                    ax.set_title(f'Frame:{frame_idx}-----ADE:{ade},FDE:{fde}')
                    # 隐藏刻度
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # 保存
                    # 动态生成文件名

                    filename = f'point_{frame_idx}.png'
                    save_path = 'F:\\sttode_groupnet3_plot_figure'
                    # 保存图表为PNG文件
                    save_path = save_path + '\\' + args.dataset + '\\'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    save_file = save_path + filename
                    if not os.path.exists(save_file):
                        plt.savefig(save_file)
                    plt.show()

            else:  ###绘制多帧，并且会存到本地文件夹
                #################################################################正常版本
                X_obs = []
                Y_obs = []
                for i in range(n):
                    x_obs = obs_traj[i][0, :]
                    y_obs = obs_traj[i][1, :]
                    # x_obs,y_obs = mat_trans_realworld(x_obs,y_obs)
                    X_obs.append(x_obs)
                    Y_obs.append(y_obs)
                X, Y = [], []

                for i in range(n):
                    X_i = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]
                    Y_i = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]
                    for j in range(20):
                        x = agent_traj[i][j, :, 0]
                        y = agent_traj[i][j, :, 1]
                        ##########################画多条曲线，预测曲线绘制多条
                        X.append(x)
                        Y.append(y)

                        #########################画单条曲线预测曲线
                        X_i += x
                        Y_i += y

                    X_i /= 20
                    Y_i /= 20
                    X.append(X_i)
                    Y.append(Y_i)
                    ###########################################

                X_GT = []
                Y_GT = []

                for i in range(n):
                    x_GT = traj_gt[i][:, 0]
                    y_GT = traj_gt[i][:, 1]
                    # x_GT, y_GT = mat_trans_realworld(x_GT, y_GT)
                    X_GT.append(x_GT)
                    Y_GT.append(y_GT)
                #################################################################正常版本

                ################################ 创建一个散点图
                rgb_color = (220 / 255, 207 / 255, 46 / 255)  # 转换后的RGB颜色值
                if args.dataset == 'hotel':
                    ax.scatter(Y, X, color=rgb_color, marker='o', s=5)
                    ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10)
                    ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10)

                    ax.imshow(bg_img, extent=[-10, 5, 5.8, -7],
                              aspect='auto')  # ax.imshow(bg_img,extent=[-10, 4, 5.8, -7], aspect='auto')右边太往右了
                if args.dataset == 'eth':
                    ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
                    ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10, label='gt')
                    ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10, label='obs')

                    ax.imshow(bg_img, extent=[-9, 20, 12.5, -3], aspect='auto')
                if args.dataset == 'zara1':
                    ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                    ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                    ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)

                    ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
                if args.dataset == 'univ':
                    ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                    ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                    ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)

                    ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
                ax.set_title(f'Frame:{frame_idx}-----ADE:{ade},FDE:{fde}')
                # 隐藏刻度
                ax.set_xticks([])
                ax.set_yticks([])

                # 保存
                # 动态生成文件名

                filename = f'point_{frame_idx}.png'
                save_path = 'F:\\sttode_plot_figure'
                # 保存图表为PNG文件
                save_path = save_path + '\\' + args.dataset + '\\'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_file = save_path + filename
                if not os.path.exists(save_file):
                    plt.savefig(save_file)
                plt.show()

                ###################################
        # miss_sample_num = count_miss_samples(agent_traj, traj_gt)
        # total_cnt += sample_motion.shape[0]
        # miss_cnt += miss_sample_num

    # miss_rate = float(miss_cnt) / float(total_cnt)
    print('Infer time :', inference_time / num_frame, '\n',
          'All Infer time:', inference_time, '\n',
          'Num_frame:', num_frame, '\n',
          'total_num_agents', total_num_agents, '\n',
          'per_agent', inference_time / total_num_agents, '\n',
          )

    return ade_meter, fde_meter  # , miss_rate



def test_model_all(model ,test_loader, args):

    total_num_pred = 0
    all_num = 0
    l2error_overall = 0
    l2error_dest = 0
    l2error_avg_04s = 0
    l2error_dest_04s = 0
    l2error_avg_08s = 0
    l2error_dest_08s = 0
    l2error_avg_12s = 0
    l2error_dest_12s = 0
    l2error_avg_16s = 0
    l2error_dest_16s = 0
    l2error_avg_20s = 0
    l2error_dest_20s = 0
    l2error_avg_24s = 0
    l2error_dest_24s = 0
    l2error_avg_28s = 0
    l2error_dest_28s = 0
    l2error_avg_32s = 0
    l2error_dest_32s = 0
    l2error_avg_36s = 0
    l2error_dest_36s = 0

    for data in test_loader:
        future_traj = np.array(data['future_traj']) * args.traj_scale # B,N,T,2
        with torch.no_grad():
            model.set_data_nba(data)
            prediction = model.inference(data)
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu()) #(BN,20,T,2)
        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]

        y = np.reshape(future_traj,(batch*actor_num,args.future_length, 2))
        y = y[None].repeat(20,axis=0)
        l2error_avg_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:1,:] - prediction[:,:,:1,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,0:1,:] - prediction[:,:,0:1,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:2,:] - prediction[:,:,:2,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,1:2,:] - prediction[:,:,1:2,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:3,:] - prediction[:,:,:3,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,2:3,:] - prediction[:,:,2:3,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:4,:] - prediction[:,:,:4,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,3:4,:] - prediction[:,:,3:4,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:5,:] - prediction[:,:,:5,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,4:5,:] - prediction[:,:,4:5,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:6,:] - prediction[:,:,:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,5:6,:] - prediction[:,:,5:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:7,:] - prediction[:,:,:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,6:7,:] - prediction[:,:,6:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:8,:] - prediction[:,:,:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,7:8,:] - prediction[:,:,7:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:9,:] - prediction[:,:,:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,8:9,:] - prediction[:,:,8:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_overall += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:10,:] - prediction[:,:,:10,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,9:10,:] - prediction[:,:,9:10,:], axis = 3),axis=2),axis=0))*batch
        all_num += batch

    print(all_num)
    l2error_overall /= all_num
    l2error_dest /= all_num

    l2error_avg_04s /= all_num
    l2error_dest_04s /= all_num
    l2error_avg_08s /= all_num
    l2error_dest_08s /= all_num
    l2error_avg_12s /= all_num
    l2error_dest_12s /= all_num
    l2error_avg_16s /= all_num
    l2error_dest_16s /= all_num
    l2error_avg_20s /= all_num
    l2error_dest_20s /= all_num
    l2error_avg_24s /= all_num
    l2error_dest_24s /= all_num
    l2error_avg_28s /= all_num
    l2error_dest_28s /= all_num
    l2error_avg_32s /= all_num
    l2error_dest_32s /= all_num
    l2error_avg_36s /= all_num
    l2error_dest_36s /= all_num
    print('##################')
    print('ADE 1.0s:',(l2error_avg_08s+l2error_avg_12s)/2)
    print('ADE 2.0s:',l2error_avg_20s)
    print('ADE 3.0s:',(l2error_avg_32s+l2error_avg_28s)/2)
    print('ADE 4.0s:',l2error_overall)

    print('FDE 1.0s:',(l2error_dest_08s+l2error_dest_12s)/2)
    print('FDE 2.0s:',l2error_dest_20s)
    print('FDE 3.0s:',(l2error_dest_28s+l2error_dest_32s)/2)
    print('FDE 4.0s:',l2error_dest)
    print('##################')

    return

def main():

    args = parser.parse_args()
    if args.dataset == 'nba':
        pass
    else:
        args.past_length = 8
        args.future_length = 12
    """ setup """
    # names = [x for x in args.model_names.split(',')]

    vae_dir = './' + args.model_save_dir + args.dataset
    all_vae_models = os.listdir(vae_dir)

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device(
        'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)

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
            args.traj_scale = args.sdd_scale
            dset_test = SDD_Dataset(
                data_set + 'test/',
                obs_len=args.past_length,
                pred_len=args.future_length,
                skip=1, traj_scale= args.sdd_scale)
        else:
            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=args.past_length,
                pred_len=args.future_length,
                skip=1, traj_scale=traj_scale)

        test_loader = DataLoader(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=0)
    # for name in names:
    best_ade = 1000
    best_fde = 1000
    best_epoch = 0
    for i in all_vae_models[-2:]:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        """ model """
        model_save_dir = args.model_save_dir + args.dataset + '/'
        # saved_path = os.path.join(model_save_dir,str(name)+'.p')
        saved_path = os.path.join(model_save_dir, i)
        print('load model from:', saved_path)
        checkpoint = torch.load(saved_path, map_location='cpu')
        training_args = checkpoint['model_cfg']

        model = STTODENet(training_args, device)
        model.set_device(device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)

        # if args.dataset =='nba':
        #     if args.vis:
        #         vis_result(test_loader, args)
        if args.dataset == 'nba':
            test_model_all(model, test_loader, args)
        else:
            print('Evaluating...')
            # traj_scale = args.traj_scale
            ade_meter, fde_meter = eval(args, test_loader, model, args.traj_scale)
            print('-' * 20 + ' STATS ' + '-' * 20)
            print('ADE: %.4f' % ade_meter.avg)
            print('FDE: %.4f' % fde_meter.avg)
            if ade_meter.avg<best_ade:
                best_ade = ade_meter.avg
                best_fde = fde_meter.avg
                best_epoch = i
    print('-' * 30 + '      Final_result    ' + '-' * 30)
    print('-' * 30 + ' The best result ever ' + '-' * 30)
    print('Best_epoch:',best_epoch)
    print('ADE: %.4f' % best_ade)
    print('FDE: %.4f' % best_fde)


if __name__ == '__main__':
    main()
