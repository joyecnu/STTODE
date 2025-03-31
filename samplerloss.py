from torch.nn import functional as F


def compute_z_kld(q_z_dist_dlow, p_z_dist_infer, agent_num, min_clip, weight):
    loss_unweighted = q_z_dist_dlow.kl(p_z_dist_infer).sum()
    loss_unweighted /= agent_num
    loss_unweighted = loss_unweighted.clamp_min_(min_clip)
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def diversity_loss(infer_dec_motion, agent_num, weight, scale):
    loss_unweighted = 0
    fut_motions = infer_dec_motion.view(*infer_dec_motion.shape[:2], -1)
    for motion in fut_motions:
        dist = F.pdist(motion, 2) ** 2
        loss_unweighted += (-dist / scale).exp().mean()
    loss_unweighted /= agent_num
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def recon_loss(fut_motion_orig, infer_dec_motion, fut_mask, weight):
    diff = infer_dec_motion - fut_motion_orig.unsqueeze(1)
    mask = fut_mask.unsqueeze(1).unsqueeze(-1)
    diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    loss_unweighted = loss_unweighted.mean()
    loss = loss_unweighted * weight
    return loss, loss_unweighted

def calculate_loss_recover(pred,target,batch_size):
    diff = target.unsqueeze(1) - pred
    avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss = avg_dist.min(dim=1)[0]
    loss = loss.mean()
    return loss
def calculate_loss_kl(qz_distribution,pz_distribution,batch_size,agent_num,min_clip):
    loss = qz_distribution.kl(pz_distribution).sum()
    loss /= (batch_size * agent_num)
    loss_clamp = loss.clamp_min_(min_clip)
    return loss_clamp

def compute_sampler_loss(args, fut_motion_orig, infer_dec_motion, batch_size,fut_mask, p_z_dist, q_z_dist_dlow, div_cfg):
    agent_num = fut_motion_orig.shape[0]
    kld_loss, kld_loss_uw = compute_z_kld(q_z_dist_dlow, p_z_dist, agent_num, args.kld_min_clamp, args.kld_weight)
    # kld_loss = 0
    # kld_loss_uw = 0
    # kld_loss = calculate_loss_kl(q_z_dist_dlow, p_z_dist, batch_size, agent_num,2.0)
    div_loss, div_loss_uw = diversity_loss(infer_dec_motion, agent_num, div_cfg['weight'], div_cfg['scale'])
    # rec_loss, rec_loss_uw = recon_loss(fut_motion_orig, infer_dec_motion, fut_mask, args.recon_weight)
    rec_loss_uw = 0
    # rec_loss = calculate_loss_recover(infer_dec_motion,fut_motion_orig, batch_size)
    total_loss = kld_loss + div_loss #+ rec_loss_uw
    loss_dict = {'kld': kld_loss, 'diverse': div_loss, 'recon': rec_loss_uw}
    loss_dict_uw ={'kld': kld_loss, 'diverse': div_loss, 'recon': rec_loss_uw}
    return total_loss, loss_dict, loss_dict_uw


def compute_sampler_loss_nba(args, fut_motion_orig, infer_dec_motion,batch_size, p_z_dist, q_z_dist_dlow, div_cfg):
    agent_num = fut_motion_orig.shape[0]
    kld_loss, kld_loss_uw = compute_z_kld(q_z_dist_dlow, p_z_dist, agent_num, args.kld_min_clamp, args.kld_weight)
    # kld_loss = 0
    # kld_loss_uw = 0
    # kld_loss = calculate_loss_kl(q_z_dist_dlow, p_z_dist, batch_size, agent_num,2.0)
    div_loss, div_loss_uw = diversity_loss(infer_dec_motion, agent_num, div_cfg['weight'], div_cfg['scale'])
    # rec_loss, rec_loss_uw = recon_loss(fut_motion_orig, infer_dec_motion, fut_mask, args.recon_weight)
    rec_loss_uw = 0
    # rec_loss = calculate_loss_recover(infer_dec_motion,fut_motion_orig, batch_size)
    total_loss = kld_loss + div_loss #+ rec_loss_uw
    loss_dict = {'kld': kld_loss, 'diverse': div_loss, 'recon': rec_loss_uw}
    loss_dict_uw ={'kld': kld_loss, 'diverse': div_loss, 'recon': rec_loss_uw}
    return total_loss, loss_dict, loss_dict_uw
