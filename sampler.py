
from utils.torchutils import *
from utils.mlp import MLP
from utils.dist import *


class Sampler(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = torch.device('cpu')
        self.args = args
        self.nk = args.sample_k
        self.nz = args.nz
        self.share_eps = args.share_eps
        self.train_w_mean = args.train_w_mean

        self.pred_model_dim = 64

        # Q net
        self.qnet_mlp = args.qnet_mlp
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)

        self.q_c = nn.Linear(self.nk * self.nz,  self.nz)
        self.linear = nn.Linear(128,64)
    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, net,  mean=True, need_weights=False):

        agent_num = net.agent_num

        net.encode_history()
        net.fu_encoder()

        history_enc = self.linear(net.past_feature)
        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)
                eps = eps.repeat((agent_num * self.nk, 1))
            else:
                eps = torch.randn([agent_num, self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=0)

        qnet_h = self.q_mlp(history_enc)
        A = self.q_A(qnet_h).view(-1, self.nz)
        b = self.q_b(qnet_h).view(-1, self.nz)

        z = b if mean else A * eps + b
        z = self.q_c(z.view(-1,self.nk* self.nz)).view(-1, self.nz)
        logvar = (A ** 2 + 1e-8).log()
        sampler_dist = Normal(mu=b, logvar=logvar)

        net.decoder_future_0(z)

        p_z_s = b if mean else A * eps + b

        net.decoder_future_1(p_z_s)

        vae_dist = net.pz_dis
        if self.args.dataset=='nba':
            dec_motion = net.diverse_pred_traj
        else:
            self.scene_orig = net.scene_orig
            dec_motion = net.diverse_pred_traj + self.scene_orig

        attn_weights = net.pred_traj


        return dec_motion, sampler_dist, vae_dist, attn_weights

    def step_annealer(self):
        pass