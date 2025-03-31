import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from utils.utils import prepare_seed, AverageMeter
from utils.dataloader import TrajectoryDataset
from model import STTODE as STODE
from vaeloss import compute_vae_loss
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter
# from eval import masked_mae_np, masked_mape_np, masked_rmse_np
from utils.metrics import compute_ADE, compute_FDE, count_miss_samples
from test import test


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

# task setting  args.start_test
parser.add_argument('--start_test', type=int, default=100)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--sdd_scale', type=float, default=50.0)

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

# parser.add_argument('--cross_range', type=int, default=2)
# parser.add_argument('--num_conv_layer', type=int, default=7)

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=3)
parser.add_argument('--asconv_layer_num', type=int, default=3)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--sample_k', type=int, default=20)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)

# loss config
parser.add_argument('--mse_weight', type=float, default=1.0)
parser.add_argument('--kld_weight', type=float, default=1.0)
parser.add_argument('--kld_min_clamp', type=float, default=2.0)
parser.add_argument('--var_weight', type=float, default=1.0)
parser.add_argument('--var_k', type=int, default=20)

# training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default='step')

parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--lr_fix_epochs', type=int, default=10)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=500)


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def print_log(dataset, epoch, total_epoch, index, total_samples, seq_name, frame, loss_str):
    # form a string and adjust format
    print_str = '{} | Epo: {:02d}/{:02d}, It: {:04d}/{:04d}, seq: {:s}, frame {:05d}, {}' \
        .format(dataset + ' vae', epoch, total_epoch, index, total_samples, str(seq_name), int(frame), loss_str)
    print(print_str)


@torch.no_grad()
def eval(loader_test, model, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()

    num_frame = 0
    inference_time = 0
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
            dec_motion, _ = model.forward()  # [ T N sn 2]  # testing function
            dec_motion = dec_motion[:,:,:,:2].permute(1, 2, 0, 3)  # [N sn T 2]
            end_time = time.time()  # 获取结束时间
            inference_time = end_time - start_time
        dec_motion = dec_motion * traj_scale
        traj_gt = pred_traj_gt.transpose(1, 2) * traj_scale  # [N 2 T] -> [N T 2]

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


    print('Infer time :', inference_time / num_frame, '\n', 'All Infer time:', inference_time, '\n', 'Num_frame:',
          num_frame)

    return ade_meter, fde_meter  # , miss_rate


def train(args, epoch, model, optimizer,  scheduler, loader_train):
    train_loss_meter = {'mse': AverageMeter(), 'kld': AverageMeter(),
                        'sample': AverageMeter(), 'total_loss': AverageMeter()}
    data_index = 0
    for cnt, batch in enumerate(loader_train):
        seq_name = batch.pop()[0]
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

        model.set_data(batch, obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)  # [T N or N*sn 2]

        pred, pred_infer,targets,fut_mask = model.forward()
        optimizer.zero_grad()
        pred       = pred.permute(1,0,2)#[N,T,2]
        pred_infer = pred_infer.permute(1, 2, 0,3)#[N,sn,T,2]
        q_z_dist= []
        p_z_dist= []
        total_loss, loss_dict, loss_dict_uw = compute_vae_loss(args, targets, pred, pred_infer, fut_mask,q_z_dist,p_z_dist)

        total_loss.backward()  # total loss is weighted
        optimizer.step()

        # save loss
        train_loss_meter['total_loss'].update(total_loss.item())
        for key in loss_dict_uw.keys():
            train_loss_meter[key].update(loss_dict_uw[key])  # printed loss item from loss_dict_uw

        # print loss
        if cnt - data_index == args.print_freq:
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            print_log(args.dataset, epoch, args.num_epochs, cnt, len(loader_train), seq_name, frame_idx, losses_str)
            data_index = cnt

    scheduler.step()
    model.step_annealer()


def main(args):
    data_set = './dataset/' + args.dataset + '/'

    prepare_seed(args.seed)
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

    if args.dataset == 'sdd':
        traj_scale = args.sdd_scale
        dset_train = SDD_Dataset(
            data_set + 'train/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)
    else:
        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    ''' === set model === '''
    net = STODE(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler_type = args.scheduler
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=args.lr_fix_epochs, nepoch=args.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=args.decay_step, decay_gamma=args.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    checkpoint_dir = './checkpoints/' + args.dataset + '/vae/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # vae_path = os.path.join(checkpoint_dir, x)
    vae_dir = './checkpoints/' + args.dataset + '/vae/'
    all_vae_models = os.listdir(vae_dir)
    if len(all_vae_models) == 0:
        print('VAE model not found!')

    for x in all_vae_models[-1:]:
        if args.dataset == 'hotel':
            x = x  # 'model_%04d.p' % 102#    45  # 90#65     #   102  #90:101
        elif args.dataset == 'eth':
            x = x  # 'model_%04d.p' % 106 # 45 100 106
        elif args.dataset == 'zara1':
            x = x  # 'model_%04d.p' % 147 # zara 60  #hotel 75 55   #141    115   133
        elif args.dataset == 'zara2':
            x = x  # 'model_%04d.p' % 80  # zara 60  #hotel 75
        elif args.dataset == 'univ':
            x = x  # 'model_%04d.p' % 60 #72  60  45
        else:
            x = x  # 'model_%04d.p' % 100  # sdd 100
        # if default_vae_model not in all_vae_models:
        #     x = all_vae_models[-1]
        vae_path = os.path.join(vae_dir, x)
        print('loading model from checkpoint: %s' % vae_path)
        model_cp = torch.load(vae_path, map_location='cpu')
    if len(all_vae_models) != 0:
        net.load_state_dict(model_cp)

    net.set_device(device)

    net.train()
    # best_valid_ade = 1000
    ################显示当前模型的参数量
    # model_structure(net)
    # if args.test:# run testing
    #     print('Evaluating...')
    #     ade_meter, fde_meter = eval(loader_test=loader_test, model=net, traj_scale=traj_scale)
    #     print('-' * 20 + ' STATS ' + '-' * 20)
    #     print('ADE: %.4f' % ade_meter.avg)
    #     print('FDE: %.4f' % fde_meter.avg)
    #     if ade_meter.avg < best_valid_ade:
    #         best_valid_ade = ade_meter.avg
    #         print('New best results!')
            # torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')
    #加载已经训练的模型
    dir = './checkpoints/' + args.dataset + '/vae/'
    x = os.listdir(dir)
    startepoch = 0
    if len(x) != 0:
        path = os.path.join(dir, x[-1])
        startepoch = int(path.split('_')[-1].split('.')[-2])


    for epoch in range(startepoch,args.num_epochs):
        if args.train:
            train(args, epoch, net, optimizer,  scheduler, loader_train)
            cp_path = os.path.join(checkpoint_dir, 'model_%04d.p') % (epoch + 1)
            model_cp = net.state_dict()
            torch.save(model_cp, cp_path)
        if epoch>args.start_test:
            test()



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
