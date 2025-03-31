from model.utils import initialize_weights
from .utils import  MLP
from hypertransformer import TransformerDecoderLayer,TransformerEncoderLayer,TransformerEncoder
from ode_demo import *

def rotation_2d_torch(x, theta, origin=None):
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin  # treat origin as center?
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x

class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''
    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()

        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)

        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim,  past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        self.init_parameters()


    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)


    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)
        
        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)

        input_feat = torch.cat((f, state_past), dim=1)


        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        
        return x_hat_after, y_hat

class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())        

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a,1,1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N,T,D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N,T,D)

class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)
            
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim*args.past_length,self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim+3,self.model_dim)


        self.nhead = 8
        self.ff_dim = 1024
        self.dropout = 0.
        self.nlayer = 1
        encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.ODE_Encoder = ODEG_Encoder(encoder_layers, self.nlayer,12)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)

        category[0:int((N-1) / 2)]
        category[0:N-1-int((N-1) / 2)]
        category[N-1, 2] = 1

        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x
    def get_agent_mask(self,agent_mask):
        self.agent_mask = agent_mask

    def forward(self, inputs,batch_size, agent_num):
        length = inputs.shape[1]

        tf_in = self.input_fc(inputs).view(batch_size*agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
  
        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, length*self.model_dim))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))


        ftraj_input1 = ftraj_input.unsqueeze(2)

        mem_agent_mask = self.agent_mask.clone()
        ftraj_inter1  = self.ODE_Encoder(ftraj_input1, mask=mem_agent_mask, num_agent=agent_num)
        ftraj_inter1 = ftraj_inter1.squeeze(2)


        final_feature = torch.cat((ftraj_input, ftraj_inter1), dim=-1)

        output_feature = final_feature.view(batch_size*agent_num,-1)
        return output_feature

class FutureEncoder(nn.Module):
    def __init__(self, args,in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        scale_num = 2 + len(self.args.hyper_scales)
        self.input_fc2 = nn.Linear(self.model_dim*self.args.future_length, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim+3, self.model_dim)

        self.nhead = 8
        self.ff_dim = 1024
        self.dropout = 0.
        self.nlayer =1
        encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.ODE_Encoder = ODEG_Encoder(encoder_layers, self.nlayer,12)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

        self.out_mlp = MLP2(scale_num*self.model_dim, [128], 'relu')#nba
        # self.out_mlp = MLP2(scale_num  * self.model_dim, [128], 'relu')
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        initialize_weights(self.qz_layer.modules())

    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:int((N - 1) / 2)]
        category[0:N - 1 - int((N - 1) / 2)]
        category[N - 1, 2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x
    def get_agent_mask(self,agent_mask):
        self.agent_mask = agent_mask

    def forward(self, inputs, batch_size,agent_num,past_feature):
        length = inputs.shape[1]
        # agent_num = 11
        tf_in = self.input_fc(inputs).view(batch_size*agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)

        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, -1))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))

        ftraj_input1 = ftraj_input.unsqueeze(2)
        mem_agent_mask = self.agent_mask.clone()

        ftraj_inter1 = self.ODE_Encoder(ftraj_input1, mask=mem_agent_mask, num_agent=agent_num)
        ftraj_inter1 = ftraj_inter1.squeeze(2)

        final_feature = torch.cat((ftraj_input, ftraj_inter1), dim=-1)

        final_feature = final_feature.view(batch_size*agent_num,-1)

        h = torch.cat((past_feature,final_feature),dim=-1)
        h = self.out_mlp(h)
        q_z_params = self.qz_layer(h)
        return q_z_params

class Decoder(nn.Module):
    def __init__(self, args):
        '''
        stolen from https://github.com/MediaBrain-SJTU/GroupNet
        '''
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.decode_way = 'RES'
        scale_num = 2
        
        self.num_decompose = args.num_decompose
        input_dim = scale_num*self.model_dim+self.args.zdim
        self.past_length = self.args.past_length
        self.future_length = self.args.future_length

        self.decompose = nn.ModuleList([DecomposeBlock(self.args.past_length, self.args.future_length, input_dim) for _ in range(self.num_decompose)])

    def forward(self, past_feature, z, batch_size_curr, agent_num_perscene, past_traj, cur_location, sample_num, mode='train'):
        agent_num = batch_size_curr * agent_num_perscene
        past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
        past_feature = past_feature.view(-1,sample_num,past_feature.shape[-1])

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = torch.cat((past_feature,z_in),dim=-1)
        hidden = hidden.view(agent_num*sample_num,-1)
        x_true = past_traj_repeat.clone()

        x_hat = torch.zeros_like(x_true)
        batch_size = x_true.size(0)
        prediction = torch.zeros((batch_size, self.future_length, 2)).cuda()
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).cuda()

        for i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
            prediction += y_hat
            reconstruction += x_hat
        norm_seq = prediction.view(agent_num*sample_num,self.future_length,2)
        recover_pre_seq = reconstruction.view(agent_num*sample_num,self.past_length,2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat
        if mode == 'inference':
            out_seq = out_seq.view(-1,sample_num,*out_seq.shape[1:])
        return out_seq,recover_pre_seq
        
class STTODENet(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args
        self.max_train_agent = args.max_train_agent
        self.rand_rot_scene = args.rand_rot_scene
        self.discrete_rot = args.discrete_rot
        # models
        scale_num = 2 + len(self.args.hyper_scales)
        self.past_encoder = PastEncoder(args)
        self.pz_layer = nn.Linear(scale_num*self.args.hidden_dim, 2 * self.args.zdim)
        if args.learn_prior:
            initialize_weights(self.pz_layer.modules())
        self.future_encoder = FutureEncoder(args)
        self.decoder = Decoder(args)
        self.param_annealers = nn.ModuleList()

    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def calculate_loss_pred(self,pred,target,batch_size):
        loss = (target-pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss
    
    def calculate_loss_kl(self,qz_distribution,pz_distribution,batch_size,agent_num,min_clip):
        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_size * agent_num)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def calculate_loss_recover(self,pred,target,batch_size):
        loss = (target-pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss
    
    def calculate_loss_diverse(self,pred,target,batch_size):
        diff = target.unsqueeze(1) - pred
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss = avg_dist.min(dim=1)[0]
        loss = loss.mean() 
        return loss

    def set_data(self, batch, pre_motion, fut_motion, pre_motion_mask, fut_motion_mask):
        device = self.device
        self.batch_size = 1
        fut_motion_orig = fut_motion.transpose(1, 2)

        pre_motion = pre_motion.permute(2, 0, 1)
        fut_motion = fut_motion.permute(2, 0, 1)

        if self.training and pre_motion.shape[1] > self.max_train_agent:
            ind = np.random.choice(pre_motion.shape[1], self.max_train_agent).tolist()
            ind = torch.tensor(ind).to(device)

            pre_motion = torch.index_select(pre_motion, 1, ind).contiguous()
            pre_motion_mask = torch.index_select(pre_motion_mask, 0, ind).contiguous()  # [N T]
            fut_motion = torch.index_select(fut_motion, 1, ind).contiguous()
            fut_motion_mask = torch.index_select(fut_motion_mask, 0, ind).contiguous()  # [N T]
            fut_motion_orig = torch.index_select(fut_motion_orig, 0, ind).contiguous()

        self.agent_num = pre_motion.shape[1]

        self.scene_orig = pre_motion[[-1]].view(-1, 2).mean(dim=0)  # [T N 2] -> [1 N 2]

        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2  # true branch
            pre_motion, pre_motion_scene_norm = rotation_2d_torch(pre_motion, theta, self.scene_orig)
            fut_motion, fut_motion_scene_norm = rotation_2d_torch(fut_motion, theta, self.scene_orig)
            fut_motion_orig, fut_motion_orig_scene_norm = rotation_2d_torch(fut_motion_orig, theta, self.scene_orig)
        else:
            theta = torch.zeros(1).to(device)
            pre_motion_scene_norm = pre_motion - self.scene_orig
            fut_motion_scene_norm = fut_motion - self.scene_orig

        pre_vel = pre_motion[1:] - pre_motion[:-1, :]
        pre_vel = torch.cat([pre_vel[[0]], pre_vel], dim=0)
        fut_vel = fut_motion - torch.cat([pre_motion[[-1]], fut_motion[:-1, :]])

        cur_motion = pre_motion[[-1]][0]
        mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        agent_mask = mask  # [N N]

        # assign values
        self.agent_mask = agent_mask

        self.pre_motion_mask = pre_motion_mask
        self.fut_motion_mask = fut_motion_mask

        self.pre_motion = pre_motion
        self.pre_vel = pre_vel
        self.pre_motion_scene_norm = pre_motion_scene_norm

        self.fut_motion = fut_motion
        self.fut_vel = fut_vel
        self.fut_motion_scene_norm = fut_motion_scene_norm

        self.fut_motion_orig = fut_motion_orig

        self.inputs = torch.cat([self.pre_motion_scene_norm, self.pre_vel], dim=-1).permute(1,0,2)  #####历史轨迹
        self.inputs_for_posterior = torch.cat([self.fut_motion_scene_norm, self.fut_vel], dim=-1).permute(1,0,2)  #####未来预测轨迹

        self.past_traj = pre_motion_scene_norm.permute(1,0,2)
        self.future_traj = fut_motion_scene_norm.permute(1,0,2)
        self.cur_location = self.past_traj[:, [-1]]

    def set_data_nba(self,data):
        device = self.device
        self.data = data
        self.batch_size = data['past_traj'].shape[0]

        self.agent_num = data['past_traj'].shape[1]

        self.past_traj = data['past_traj'].view(self.batch_size * self.agent_num, self.args.past_length, 2).to(device).contiguous()
        self.future_traj = data['future_traj'].view(self.batch_size * self.agent_num, self.args.future_length, 2).to(
            device).contiguous()
        self.scene_orig = self.past_traj
        self.past_vel = self.past_traj[:, 1:] - self.past_traj[:, :-1, :]
        self.past_vel = torch.cat([self.past_vel[:, [0]], self.past_vel], dim=1)

        self.future_vel = self.future_traj - torch.cat([self.past_traj[:, [-1]], self.future_traj[:, :-1, :]], dim=1)
        self.cur_location = self.past_traj[:, [-1]]

        self.inputs = torch.cat((self.past_traj, self.past_vel), dim=-1)
        self.inputs_for_posterior = torch.cat((self.future_traj, self.future_vel), dim=-1)


        mask = torch.zeros([self.inputs.shape[0], self.inputs.shape[0]]).to(device)
        agent_mask = mask  # [N N]
        self.agent_mask = agent_mask
    
    def encode_history(self):

        if(self.args.dataset =='nba'):
            self.future_encoder.get_agent_mask(self.agent_mask)
            self.past_encoder.get_agent_mask(self.agent_mask)
        else:
            self.future_encoder.get_agent_mask(self.pre_motion_mask)
            self.past_encoder.get_agent_mask(self.fut_motion_mask)
        self.past_feature = self.past_encoder(self.inputs, self.batch_size, self.agent_num)

    def fu_encoder(self):
        self.qz_param = self.future_encoder(self.inputs_for_posterior, self.batch_size, self.agent_num, self.past_feature)
        #
        ### q dist ###
        sample_num = 20
        qz_param_repeat = self.qz_param.repeat_interleave(sample_num, dim=0)
        if self.args.ztype == 'gaussian':
            self.qz_dis = Normal(params=qz_param_repeat)
            self.qz_distribution = Normal(params=self.qz_param)
        else:
            ValueError('Unknown hidden distribution!')
        self.qz_sampled = self.qz_distribution.rsample()

        ### p dist ###
        if self.args.learn_prior:
            pz_param = self.pz_layer(self.past_feature)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=pz_param)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(self.past_feature.shape[0], self.args.zdim).to(self.past_traj.device),
                                        logvar=torch.zeros(self.past_feature.shape[0], self.args.zdim).to(self.past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')
        self.pz_distribution = pz_distribution
        self.pz_sampled = self.pz_distribution.rsample()

        # self.pz_dis = pz_distribution

    def decoder_future_1(self,pz_sampled):
        
        self.diverse_pred_traj,self.attn_weights = self.decoder(self.past_feature_repeat,pz_sampled,self.batch_size,self.agent_num,
                                       self.past_traj,self.cur_location,sample_num=20,mode='inference')

    def decoder_future_0(self, qz_sampled):
        self.pred_traj, self.recover_traj = self.decoder(self.past_feature, qz_sampled, self.batch_size, self.agent_num,
                                                     self.past_traj, self.cur_location, sample_num=1)

        ### p dist for best 20 loss ###
        sample_num = 20
    
        past_feature_repeat = self.past_feature.repeat_interleave(sample_num, dim=0)
        if self.args.ztype == 'gaussian':
            pz_distribution = Normal(
                mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(self.past_traj.device),
                logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(self.past_traj.device))
        else:
            ValueError('Unknown hidden distribution!')
        self.past_feature_repeat = past_feature_repeat
        self.pz_sampled = pz_distribution.rsample()

        self.pz_dis = pz_distribution

    def forward(self):
        self.encode_history()
        self.fu_encoder()

        self.decoder_future_0(self.qz_sampled)
        loss_pred = self.calculate_loss_pred(self.pred_traj,self.future_traj,self.batch_size)

        loss_recover = self.calculate_loss_recover(self.recover_traj,self.past_traj,self.batch_size)
        loss_kl = self.calculate_loss_kl(self.qz_distribution,self.pz_distribution,self.batch_size,self.agent_num,self.args.min_clip)

        self.decoder_future_1(self.pz_sampled)

        loss_diverse = self.calculate_loss_diverse(self.diverse_pred_traj,self.future_traj,self.batch_size)
        total_loss = loss_pred + loss_recover + loss_kl+ loss_diverse

        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item()

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def inference(self, data):
        self.future_encoder.get_agent_mask(self.agent_mask)
        self.past_encoder.get_agent_mask(self.agent_mask)
        device = self.device
        if self.args.dataset == 'nba':
            batch_size = data['past_traj'].shape[0]
            agent_num = data['past_traj'].shape[1]
            past_traj = data['past_traj'].view(batch_size * agent_num, self.args.past_length, 2).to(device).contiguous()
            past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
            past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        else:
            batch_size = 1
            past_traj = self.past_traj
            past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
            past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)
            agent_num = self.agent_num
            self.future_encoder.get_agent_mask(self.pre_motion_mask)
            self.past_encoder.get_agent_mask(self.fut_motion_mask)

        cur_location = past_traj[:,[-1]]

        inputs = torch.cat((past_traj,past_vel),dim=-1)

        past_feature = self.past_encoder(inputs,batch_size,agent_num)

        sample_num = 20
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device), 
                                        logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj,_ = self.decoder(past_feature_repeat,z,batch_size,agent_num,past_traj,cur_location,sample_num=self.args.sample_k,mode='inference')
        diverse_pred_traj = diverse_pred_traj.permute(1,0,2,3)
        if self.args.dataset != 'nba':
            diverse_pred_traj =  diverse_pred_traj + self.scene_orig
        return diverse_pred_traj
