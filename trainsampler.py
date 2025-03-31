import os
import sys
import argparse
from torch import optim
from torch.utils.data import DataLoader as dataLoader1

from utils.dataloader import TrajectoryDataset
from model.STTODE import STTODENet
from sampler import Sampler
from samplerloss import compute_sampler_loss,compute_sampler_loss_nba
from utils.sddloader import SDD_Dataset
from data.dataloader_nba import NBADataset, seq_collate
sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter

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

# vae_epoch
parser.add_argument('--vae_epoch', type=int, default=100)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

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

# loss config
parser.add_argument('--kld_weight', type=float, default=0.1)
parser.add_argument('--kld_min_clamp', type=float, default=10)
parser.add_argument('--recon_weight', type=float, default=5.0)



parser.add_argument('--lr_fix_epochs', type=int, default=10)



parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
# assign diversity loss config
def get_diversity_config(dataset):
    if dataset == 'sdd':
        weight, scale = 0.5,0.5
    elif dataset == 'eth':
        weight, scale = 1,1
    elif dataset == 'univ':
        weight, scale = 10,10.0
    elif dataset == 'nba':
        weight, scale = 1,1.0
    else:
        weight, scale = 3,2
    return {'weight': weight, 'scale': scale}


def print_log(dataset, epoch, total_epoch, index, total_samples, seq_name, frame, loss_str):
    # form a string and adjust format
    print_str = '{} | Epo: {:02d}/{:02d}, It: {:04d}/{:04d}, seq: {:s}, frame {:05d}, {}' \
        .format(dataset + ' sampler', epoch, total_epoch, index, total_samples, str(seq_name), int(frame), loss_str)
    print(print_str)


def train(args, epoch, STTODENet, sampler, optimizer, scheduler, train_loader, div_cfg):
    train_loss_meter = {'kld': AverageMeter(), 'diverse': AverageMeter(),
                        'recon': AverageMeter(), 'total_loss': AverageMeter()}
    data_index = 0
    total_iter_num = len(train_loader)
    iter_num = 0

    if args.dataset == 'nba':
        for data in train_loader:
            STTODENet.decoder.train()
            STTODENet.set_data_nba(data)
            dec_motion, sampler_dist, vae_dist, _ = sampler.forward( STTODENet)  # (STTODENet.inputs,STTODENet.batch_size,STTODENet.agent_num)  # [T N sn 2]
            batch_size = data['past_traj'].shape[0]
            agent_num = data['past_traj'].shape[1]
            device = 'cuda:0'
            future_traj = data['future_traj'].view(batch_size * agent_num, args.future_length, 2).to(
            device).contiguous()
            dec_motion =dec_motion.permute(1,0,2,3)
            fut_motion_orig = future_traj.reshape([-1,10,2])#.transpose(1, 2)  # [32 N 2 T] -> [32 N T 2]
            dec_motion = dec_motion.reshape([-1,20,10,2])#[32 N  20 T 2]
            total_loss, loss_dict, loss_dict_uw = compute_sampler_loss_nba(args, fut_motion_orig, dec_motion, 1,
                                                                       vae_dist, sampler_dist, div_cfg)
            optimizer.zero_grad()
            # sampler.train()
            total_loss.backward()
            optimizer.step()
            # STTODENet.eval()
            # save loss
            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_dict_uw.keys():
                train_loss_meter[key].update(loss_dict_uw[key])


            if iter_num % args.iternum_print == 0:
                losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
                print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} |{:s}'
                .format(epoch,args.num_epochs,iter_num,total_iter_num,losses_str))
            iter_num += 1
    else:
        for cnt, batch in enumerate(train_loader):

            STTODENet.decoder.train()
            seq_name = batch.pop()[0]
            batch = [tensor[0].cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
            non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask, frame_idx = batch

            STTODENet.set_data(batch, obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)  # [T N or N*sn 2]
            dec_motion, sampler_dist, vae_dist, _ = sampler.forward(STTODENet)

            fut_motion_orig = pred_traj_gt.transpose(1, 2)  # [N 2 T] -> [N T 2]


            total_loss, loss_dict, loss_dict_uw = compute_sampler_loss(args, fut_motion_orig, dec_motion, 1, pred_loss_mask,
                                                                       vae_dist, sampler_dist, div_cfg)
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()

            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_dict_uw.keys():
                train_loss_meter[key].update(loss_dict_uw[key])

            if cnt - data_index == args.print_freq:
                losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
                print_log(args.dataset, epoch, args.num_epochs, cnt, len(train_loader), seq_name, frame_idx, losses_str)
                data_index = cnt

    scheduler.step()
    sampler.step_annealer()


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
    if args.dataset == 'eth':

        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=args.past_length,
            pred_len=args.future_length,
            skip=1, traj_scale=traj_scale)

    elif args.dataset == 'sdd':
        traj_scale = args.sdd_scale
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
    if (1):
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


    ''' === set model === '''

    net = STTODENet(args,device)  # load STTODENet

    # load STTODENet model
    vae_dir = './saved_models/' + args.dataset + '/'
    all_vae_models = os.listdir(vae_dir)
    if len(all_vae_models) == 0:
        print('VAE model not found!')
        return

    default_vae_model = 'model_%04d.p' % args.vae_epoch
    if default_vae_model not in all_vae_models:
        default_vae_model = all_vae_models[-1]
    checkpoint_path = os.path.join(vae_dir, default_vae_model)
    print('loading model from checkpoint: %s' % checkpoint_path)


    model_load = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(model_load['model_dict'])


    sampler = Sampler(args)
    optimizer = optim.Adam(sampler.parameters(), lr=args.lr)
    scheduler_type = args.scheduler
    if scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=args.lr_fix_epochs, nepoch=args.num_epochs)
    elif scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=args.decay_step, decay_gamma=args.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    checkpoint_dir = './checkpoints/' + args.dataset + '/sampler/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    div_cfg = get_diversity_config(args.dataset)

    net.set_device(device)


    #加载已经训练的模型
    sampler_dir = './checkpoints/' + args.dataset + '/sampler/'
    x = os.listdir(sampler_dir)
    startepoch = 0
    if len(x)!=0:
        sampler_path = os.path.join(sampler_dir,x[-1])
        model_cp = torch.load(sampler_path, map_location='cpu')
        sampler.load_state_dict(model_cp)
        startepoch = int(sampler_path.split('_')[-1].split('.')[-2])


    sampler.set_device(device)
    sampler.train()

    for epoch in range(startepoch,args.num_epochs):
        train(args, epoch, net, sampler, optimizer, scheduler, train_loader, div_cfg)
        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            cp_path = os.path.join(checkpoint_dir, 'model_%04d.p') % (epoch + 1)  # need to add epoch num
            model_cp = sampler.state_dict()
            torch.save(model_cp, cp_path)
            ###
        net.decoder.eval()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

