import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time

gettime = 0
formatted_time = 0

def show_eth(args,frame_idx,traj_gt,obs_traj,agent_traj,ade=0,fde=0):
    '''
    agent_traj 预测轨迹
    obs_traj   历史轨迹
    traj_gt    未来轨迹
    frame_idx   当前帧数
    '''
################################

    if args.printfigure:  # 显示图片
        # F:\\ref_figure
        # 读取背景图片
        if args.dataset == 'hotel':
            background_image_path = 'F:\\ref_figure\\seq_hotel\\'#your path  save background pictures
        elif args.dataset == 'eth':
            background_image_path = 'F:\\ref_figure\\seq_eth\\'#your path save background pictures
        elif args.dataset == 'zara1':
            background_image_path = 'F:\\ref_figure\\crowds_zara01\\'#your path save background pictures
        elif args.dataset == 'univ':
            background_image_path = 'F:\\ref_figure\\students003\\' #your path   save background pictures
        else:
            pass

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
                elif args.dataset == 'eth':
                    fig, ax = plt.subplots(figsize=(6.40, 4.80))
                elif args.dataset == 'zara1':
                    fig, ax = plt.subplots(figsize=(7.20, 5.76))
                elif args.dataset == 'univ':
                    fig, ax = plt.subplots(figsize=(7.20, 5.76))
                else:
                    pass
            else:
                pass
        else:
            if args.dataset == 'hotel':
                # angle = 0  # 旋转角度顺时针
                # bg_img = rotate(bg_img, angle, reshape=True)
                fig, ax = plt.subplots(figsize=(7.20, 5.76))
            elif args.dataset == 'eth':
                fig, ax = plt.subplots(figsize=(6.40, 4.80))#(figsize=(6.40, 4.80))
            elif args.dataset == 'zara1':
                fig, ax = plt.subplots(figsize=(7.20, 5.76))
            elif args.dataset == 'univ':
                fig, ax = plt.subplots(figsize=(7.20, 5.76))
            else:
                pass
        # 示例数据
        n = traj_gt.shape[0]  # 当前人数
        # n=10

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


                ax.set_xticks([])
                ax.set_yticks([])

                # 保存
                # 动态生成文件名

                filename = f'point_{frame_idx}.png'
                save_path = 'F:\\sttode_groutnet6_show'
                # 保存图表为PNG文件
                save_path = save_path + '\\' + args.dataset + '\\'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_file = save_path + filename
                if not os.path.exists(save_file):
                    plt.savefig(save_file)
                plt.show()
            else:
                pass
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
                #     X_i += x
                #     Y_i += y
                #
                # X_i /= 20
                # Y_i /= 20
                # X.append(X_i)
                # Y.append(Y_i)
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

                # ax.imshow(bg_img, extent=[-10, 5, 5.8, -7],aspect='auto')  # ax.imshow(bg_img,extent=[-10, 4, 5.8, -7], aspect='auto')右边太往右了
            elif args.dataset == 'eth':
                ax.scatter(Y, X, color=rgb_color, marker='o', s=5, label='predict')
                ax.scatter(Y_GT, X_GT, color='blue', marker='o', s=10, label='gt')
                ax.scatter(Y_obs, X_obs, color='red', marker='o', s=10, label='obs')
                # ax.scatter([-y for y in Y],[-x for x in X], color=rgb_color, marker='o', s=10, label='predict')
                # ax.scatter([-y for y in Y_GT],[-x for x in X_GT],color='blue', marker='o', s=2, label='gt')
                # ax.scatter([-y for y in Y_obs],[-x for x in X_obs], color='red', marker='o', s=10, label='obs')

                # ax.imshow(bg_img, extent=[-9, 20, 12.5, -3], aspect='auto')
            elif args.dataset == 'zara1':
                ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)

                ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
            elif args.dataset == 'univ':
                ax.scatter(X, Y, color=rgb_color, marker='o', s=5)
                ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
                ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)

                ax.imshow(bg_img, extent=[0, 15, 0, 14], aspect='auto')
            else:
                pass
            ax.set_title(f'Frame:{frame_idx}----- ADE: {ade:.2f}, FDE: {fde:.2f}-----Num: {n}')
            # 隐藏刻度
            ax.set_xticks([])
            ax.set_yticks([])

            # 保存
            # 动态生成文件名

            filename = f'point_{frame_idx}.png'
            save_path = 'F:\\sttode_groupnet6_show'
            # 保存图表为PNG文件
            save_path = save_path + '\\' + args.dataset + '\\'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file = save_path + filename
            if not os.path.exists(save_file):
                plt.savefig(save_file)
            plt.show()

            ###################################


def show_sdd(args,frame_idx,traj_gt,obs_traj,agent_traj,ade=0,fde=0):

    wigth,heith = 2000,2000
    fig, ax = plt.subplots(figsize=(wigth/100,heith/100))

    n = traj_gt.shape[0]  # 当前人数

    obs_traj = obs_traj.detach().cpu().numpy()

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
            ########################## 画多条曲线，预测曲线绘制多条
            X.append(x)
            Y.append(y)

            #########################画单条曲线预测曲线
        #     X_i += x
        #     Y_i += y
        #
        # X_i /= 20
        # Y_i /= 20
        # X.append(X_i)
        # Y.append(Y_i)
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
    if args.dataset == 'sdd':
        for i in range(len(X)):  # marker='o'
            ax.plot(X[i][:], Y[i][:], color=rgb_color, markersize=1.5, linewidth=0.5,label='pred')
        # ax.scatter(X_GT, Y_GT, color='blue', marker='o', s=10)
        for i in range(len(X_GT)):  # marker='o'
            ax.plot(X_GT[i][:], Y_GT[i][:], color='blue', markersize=2, linewidth=1,label='gt')
        # ax.scatter(X_obs, Y_obs, color='red', marker='o', s=10)
        for i in range(len(X_obs)):  # marker='o'
            ax.plot(X_obs[i][:], Y_obs[i][:], color='red', markersize=2, linewidth=1,label='obs')

    else:
        pass
    ax.set_title(f'Frame:{frame_idx}----- ADE: {ade:.2f}, FDE: {fde:.2f}-----Num: {n}')
    # 隐藏刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 获取系统时间
    global gettime
    global formatted_time
    if gettime ==0:
        system_time = time.localtime()
        formatted_time= time.strftime("%Y_%m_%d%H_%M_%S", system_time)
        gettime+=1
    Scene_of_sdd = 'hyang_9'
    vector_file = f'point_{frame_idx}.pdf'#f'point_{Scene_of_sdd[2:-2]}.pdf'

    save_vector_path = 'F:\\sdd_figure_sttode_groupnet\\'+ formatted_time +'\\'
    save_vector      = save_vector_path + vector_file
    if not os.path.exists(save_vector_path):
        os.mkdir(save_vector_path)
    if ade>10:
        plt.savefig(save_vector, format="pdf")
    # while not os.path.exists(save_vector):
    #     print("正在等待保存完成...")
    # print("保存完成!!!")
    # plt.show()

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


def draw_result(args, future, past, mode='pre'):
    # b n t 2
    print('drawing...')
    trajs = np.concatenate((past, future), axis=2)
    batch = trajs.shape[0]
    for idx in range(50):
        plt.clf()
        traj = trajs[idx]
        traj = traj * 94 / 28
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
                points = [(traj[j, i, 0], traj[j, i, 1])]
                (x, y) = zip(*points)
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre, s=20, alpha=1)
                else:
                    plt.scatter(x, y, color=color, s=20, alpha=1)

            for i in range(length - 1):
                points = [(traj[j, i, 0], traj[j, i, 1]), (traj[j, i + 1, 0], traj[j, i + 1, 1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre, alpha=0.5, linewidth=2)
                else:
                    plt.plot(x, y, color=color, alpha=1, linewidth=2)

        court = plt.imread("datasets/nba/court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN], alpha=0.5)
        vispath = 'vis/' + args.dataset + '/'
        if not os.path.exists(vispath):
            os.makedirs(vispath)
        if mode == 'pre':
            # plt.savefig('vis/nba/'+str(idx)+'pre.png')
            plt.savefig(vispath + str(idx) + 'pre.png')
        else:
            # plt.savefig('vis/nba/'+str(idx)+'gt.png')
            plt.savefig(vispath + str(idx) + 'gt.png')
        plt.show()
    print('ok')
    return


def vis_result(test_loader, model, args):
    total_num_pred = 0
    all_num = 0

    for data in test_loader:
        future_traj = np.array(data['future_traj']) * args.traj_scale  # B,N,T,2
        model.set_data_nba(data)
        with torch.no_grad():
            prediction = model.inference(data)
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())  # (BN,20,T,2)
        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]

        y = np.reshape(future_traj, (batch * actor_num, args.future_length, 2))
        y = y[None].repeat(20, axis=0)
        error = np.mean(np.linalg.norm(y - prediction, axis=3), axis=2)
        indices = np.argmin(error, axis=0)
        best_guess = prediction[indices, np.arange(batch * actor_num)]
        best_guess = np.reshape(best_guess, (batch, actor_num, args.future_length, 2))
        gt = np.reshape(future_traj, (batch, actor_num, args.future_length, 2))
        # previous_3D = np.reshape(previous_3D,(batch,actor_num,args.future_length, 2))
        previous_3D = np.reshape(gt, (batch, actor_num, args.future_length, 2))

        draw_result(args, best_guess, previous_3D)
        draw_result(args, gt, previous_3D, mode='gt')
    return
