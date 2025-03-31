import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
from core.manifolds import Oblique
from hyptransformerlib import hyp_mhsa

import torch
from torch import nn, Tensor
import copy
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from typing import Optional  # need to pay attention



class TransformerEncoder_ode(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.src = None
        self.mask = None
        self.src_key_padding_mask  = None
        self.num_agent = 1


    def forward(self,t,src ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=self.mask, src_key_padding_mask=self.src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder_ode(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.memory=None
        self.tgt_mask= None
        self.memory_mask= None
        self.seq_mask = False
        self.tgt_key_padding_mask= None
        self.memory_key_padding_mask= None
        self.need_weights = False
        self.num_agent = 1
    def return_wight(self):
        return {'self_attn_weights': self.self_attn_weights, 'cross_attn_weights': self.cross_attn_weights}
    def forward(self, t,tgt) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        self.self_attn_weights = [None] * len(self.layers)
        self.cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, self.self_attn_weights[i], self.cross_attn_weights[i] = mod(output, self.memory, tgt_mask=self.tgt_mask,
                                                                      memory_mask=self.memory_mask, seq_mask=self.seq_mask,
                                                                      tgt_key_padding_mask=self.tgt_key_padding_mask,
                                                                      memory_key_padding_mask=self.memory_key_padding_mask,
                                                                      need_weights=self.need_weights)

        if self.norm is not None:
            output = self.norm(output)

        if self.need_weights:  # need to modify, do not stack
            self.self_attn_weights = torch.stack(self.self_attn_weights).cpu().numpy()
            self.cross_attn_weights = torch.stack(self.cross_attn_weights).cpu().numpy()

        return output

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, tgt):

        self.odefunc.tgt = tgt.clone().detach()


    def setparrament(self,memory,tgt_mask,memory_mask,seq_mask,tgt_key_padding_mask,
                     memory_key_padding_mask,
                     need_weights,
                     num_agent):
        self.odefunc.memory = memory
        self.odefunc.tgt_mask = tgt_mask,
        self.odefunc.memory_mask = memory_mask
        self.odefunc.seq_mask = seq_mask
        self.odefunc.tgt_key_padding_mask = tgt_key_padding_mask
        self.odefunc.memory_key_padding_mask = memory_key_padding_mask
        self.odefunc.need_weights = need_weights
        self.odefunc.num_agent = num_agent
    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        wight = {'self_attn_weights': self.odefunc.self_attn_weights, 'cross_attn_weights': self.odefunc.cross_attn_weights}
        return z,wight

class ODEblock_en(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock_en, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, tgt):

        self.odefunc.tgt = tgt.clone().detach()


    def setparrament(self,src, mask, src_key_padding_mask,
                num_agent=1):
        self.odefunc.src = src
        self.odefunc.mask = mask,
        self.odefunc.src_key_padding_mask = src_key_padding_mask
        self.odefunc.num_agent = num_agent
    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        # wight = {'self_attn_weights': self.odefunc.self_attn_weights, 'cross_attn_weights': self.odefunc.cross_attn_weights}
        return z



# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, decoder_layers, nlayer,  time):#decoder_layers transformer 层   nlayer transformer层数
        super(ODEG, self).__init__()

        self.odeblock = ODEblock(TransformerDecoder_ode(decoder_layers, nlayer), t=torch.tensor([0, time]))


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, seq_mask=False, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, need_weights=False, num_agent=1):
        self.odeblock.set_x0(tgt)
        self.odeblock.setparrament(memory=memory,tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   seq_mask=seq_mask,tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   need_weights=need_weights,
                                   num_agent=num_agent)
        z ,wight= self.odeblock(tgt)
        return F.relu(z),wight


# Define the ODEGCN model.
class ODEG_Encoder(nn.Module):
    def __init__(self, encoder_layer, nlayer,  time):#decoder_layers transformer 层   nlayer transformer层数
        super(ODEG_Encoder, self).__init__()
        self.odeblock = ODEblock_en(TransformerEncoder_ode(encoder_layer, nlayer), t=torch.tensor([0, time]))


    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                    num_agent=1):
        self.odeblock.set_x0(src)
        self.odeblock.setparrament(src=src,
                                   mask =mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   num_agent=num_agent)
        z = self.odeblock(src)
        return F.relu(z)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))