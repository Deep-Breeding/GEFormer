################################################## attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        #print("queries",queries.shape)
        queries = self.query_projection(queries).view(B, L, H, -1) # [24, 148, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class OD_Attention(nn.Module): # ODconv
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OD_Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv1d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv1d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv1d(attention_channel, kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv1d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x): 
        x = x.permute(0, 2, 1) 
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)



################################################## embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.float()
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2) 
        return x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='d'):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 3
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        x = x.float()
        x = self.embed(x)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type, freq='d', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): 
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)






################################################## ODconv.py
import os
import torch
import numpy as np
import torch.nn as nn
#from tools.attn import  Attention

import torch.nn.functional as F


class ODConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv1d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        Attn = OD_Attention
        self.attention = Attn(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x.permute(0, 2, 1)  
        batch_size, in_planes, length = x.size()
        x = x * channel_attention  
        x = x.reshape(1, -1, length)  
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size])
        output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv1d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)



################################################## encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):

        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns




################################################## TimeFeature_Block.py
import torch
import torch.nn as nn
#from tools.attn import ProbAttention, AttentionLayer
#from tools.embed import DataEmbedding
#from tools.encoder import Encoder, EncoderLayer, ConvLayer
#from tools.ODconv import ODConv1d
import math

class TimeFeatureBlock(nn.Module):
    def __init__(self,args2, env_days):
        super(TimeFeatureBlock, self).__init__()

        self.args2 = args2
        self.output_attention = False

        # Encoding
        self.enc_embedding = DataEmbedding(args2.enc_in, d_model=128, embed_type='timeF', freq='d', dropout=0.05)
        # Attention
        Attn = ProbAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor=5, attention_dropout=0.05, output_attention=self.output_attention), 
                                d_model=126, n_heads=6, mix=False),
                    d_model=126,
                    d_ff=2048,
                    dropout=0.05,
                    activation='gelu'
                ) for l in range(2)
            ],
            [
                ConvLayer(126) for l in range(1)
            ] ,
            norm_layer=torch.nn.LayerNorm(126)
        )
       
        self.projection = nn.Linear(126, args2.c_out, bias=True)

        self.fc = nn.Linear(math.ceil(env_days/2), 1)

        self.fc2 = nn.Sequential(

            nn.Linear(125, 76),
            nn.ReLU(),
            nn.Linear(76, 38)
        )

        self.conv_env = nn.Sequential(
            nn.Conv1d(in_channels=75, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.ODconv = ODConv1d(env_days, env_days, 3) 
        


    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.ODconv(enc_out)

        enc_out, self.attns = self.encoder(enc_out, attn_mask=enc_self_mask)
    
        enc_out = enc_out.permute(0,2,1)
        enc_out = self.fc(enc_out)
        enc_out = enc_out.permute(0,2,1)

        return enc_out 



################################################## gMLP.py
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# functions

def exists(val):
    return val is not None

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        causal = False,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
        circulant_matrix = False
    ):
        super().__init__()
        dim_out = dim // 2
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act

        # parameters

        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))

    def forward(self, x, gate_res = None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix

            dim_seq = weight.shape[-1]
            weight = F.pad(weight, (0, dim_seq), value = 0)
            weight = repeat(weight, '... n -> ... (r n)', r = dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            # give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        if self.causal:
            weight, bias = weight[:, :n, :n], bias[:, :n]
            mask = torch.ones(weight.shape[-2:], device = device).triu_(1).bool()
            mask = rearrange(mask, 'i j -> () i j')
            weight = weight.masked_fill(mask, 0.)

        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)

        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () h n ()')

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        heads = 1,
        attn_dim = None,
        causal = False,
        act = nn.Identity(),
        circulant_matrix = False
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix = circulant_matrix)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        return x

# main classes

class gMLPVision(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        snp_len,
        heads = 1,
        ff_mult = 4,
        channels = 1,
        attn_dim = None,
        prob_survival = 1.
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        image_height, image_width = pair(image_size)       
        patch_height, patch_width = pair(patch_size)      
        #assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        #num_patches = (image_height[0] // patch_height[0]) * (image_width[1] // patch_width[1])
        num_patches = 200 
        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(1*snp_len*1, dim) 
        )
        
       
        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)




################################################## CrossGated_MLP.py
import torch
import torch.nn as nn

class CrossGatedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        # MLPs for computing x1 and x2 hidden representations
        self.mlp_x1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size),
            nn.GELU()
        )
        
        self.mlp_x2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size),
            nn.GELU()
        )
        
        # MLPs for computing x1 and x2 gates
        self.gate_x1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        self.gate_x2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Compute x1 and x2 hidden representations
        hidden_x1 = self.mlp_x1(x1)
        hidden_x2 = self.mlp_x2(x2)
        
        # Compute x1 and x2 gates
        gate_x1 = self.gate_x1(x1)
        gate_x2 = self.gate_x2(x2)
        
        # Compute fused features using the cross-gated MLP mechanism
        cross_gated_x1 = (1 - gate_x1) * hidden_x1 + gate_x2 * hidden_x2
        cross_gated_x2 = (1 - gate_x2) * hidden_x2 + gate_x1 * hidden_x1
        
        # Concatenate the fused features and return
        fused_features = torch.cat([cross_gated_x1, cross_gated_x2], dim=1)
        return fused_features





################################################## mydataset.py
import pandas as pd
import torch
from torch.utils import data

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets = {
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

def time_features(dates, freq):
    dates['month'] = dates.date.apply(lambda row: row.month, 1)
    dates['day'] = dates.date.apply(lambda row: row.day, 1)
    dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)

    freq_map = {
        'd': ['month', 'day', 'weekday']
    }
    return dates[freq_map[freq.lower()]].values


class myDataset(data.Dataset):
    def __init__(self, id, phe, dictseq, dictenv): 
        self.id = id
        self.phe = phe 
        self.dictseq = list(dictseq.values())
        self.dictenv = dictenv
        

    def __getitem__(self, index):
        env_list = self.dictenv.get(self.id[index])

        self.data_stamp = env_list.iloc[:, 0]
        self.data_x = env_list.iloc[:, 1:]
 
        self.data_stamp = pd.to_datetime(self.data_stamp) 
        self.data_stamp = pd.DataFrame(self.data_stamp)
        self.data_stamp = time_features(self.data_stamp, freq='d') 
        self.data_x = torch.tensor(self.data_x.to_numpy())
        self.data_stamp = torch.tensor(self.data_stamp)
        return self.id[index], self.phe[index], self.dictseq[index], self.data_x, self.data_stamp

    def __len__(self):
        return len(self.id) 




################################################## mymodel.py
import torch
import torch.nn as nn
#from tools.gMLP import gMLPVision
#from tools.CrossGated_MLP import CrossGatedMLP
#from tools.TimeFeature_Block import TimeFeatureBlock

class GEFormer(nn.Module):
    def __init__(self, args2, snp_len, env_days):
        super(GEFormer, self).__init__()
        
        dout = args2.dropout
        dep = args2.depth
        L1 = args2.neurons1 
        L2 = args2.neurons2
        
        self.gmlp = gMLPVision(image_size = (snp_len, 1) ,
        patch_size = (snp_len, 1) ,
        num_classes = 126, 
        dim = 126,
        depth = dep,
        snp_len = snp_len
        ) 

        self.TimeFeatureBlock = TimeFeatureBlock(args2,env_days)

        self.cgMLP = CrossGatedMLP(126)

        self.fc = nn.Sequential(
            nn.Linear(756,L1),
            nn.LeakyReLU(),  
            nn.Dropout(dout),
            nn.Linear(L1, L2),
            nn.LeakyReLU(),  
            nn.Dropout(dout),
            nn.Linear(L2, 1)
        )

    def forward(self, x, x1, x2): 
        x3 = self.TimeFeatureBlock(x1,x2) 

        x3 = x3.squeeze(1)
        x = x.transpose(0, 1)
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        x = self.gmlp(x) 

        x4 = torch.mul(x, x3)

        a = self.cgMLP(x,x3)  
        b = self.cgMLP(x,x4)
        c = self.cgMLP(x3,x4)
        concatenated = torch.cat([a, b, c], dim=1)

        predict = self.fc(concatenated)
        return predict




################################################## train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader
from itertools import islice
import random
import argparse
#import subprocess
from train_optuna import op_train
import sys
from collections import OrderedDict
import ast
import copy

#from mydataset import myDataset
#from mymodel import GEFormer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = False

def geformer(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13,arg14,arg15):
    setup_seed(147)

    parser = argparse.ArgumentParser(description="Genome-wide prediction model for genotype-environment interaction.")

    parser.add_argument('--geno_path', type=str, default=arg1, help='path of geno file')
    parser.add_argument('--pheno_path', type=str, default=arg2, help='path of pheno file')
    parser.add_argument('--pheno_name', type=str, default=arg3, help='name of phenotype')
    parser.add_argument('--env_path', type=str, default=arg4, help='path of environment file')
    parser.add_argument('--CVF_path', type=str, default=arg5, help='path of cvf file')
    parser.add_argument('--model_path', type=str, default=arg6, help='path of model file')
    parser.add_argument('--device', default=arg7, help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument("--optuna", type=lambda x: x.lower() == "true", default=arg8, help='whether to adjust parameters')

    parser.add_argument('--optuna_epoch', type=int,default=arg9, help='number of attempts with different parameter combinations')
    parser.add_argument('--batch', type=int, default=arg10, help='batchsize')
    parser.add_argument('--dropout', type=float, default=arg11, help='dropout')
    parser.add_argument('--depth', type=int, default=arg12, help='depth')
    parser.add_argument('--neurons1', type=int, default=arg13, help='neurons1 number')
    parser.add_argument('--neurons2', type=int,default=arg14, help='neurons2 number')
    parser.add_argument('--lr', type=float,default=arg15, help='learning rate')

    args2 = parser.parse_args()

    if args2.optuna is True:
        print("Begin parameter optimization.")
        output = op_train(args2.device,args2.geno_path,args2.pheno_path,args2.pheno_name,args2.env_path,args2.CVF_path,args2.optuna_epoch)
        print("Parameter optimization finished.")

        args2.batch = output['batch']
        args2.depth = output['depth']
        args2.dropout = output['dropout']
        args2.lr = output['lr']
        args2.neurons1 = output['neurons1']
        args2.neurons2 = output['neurons2']
    else:
        args2.optuna_epoch = arg9
        args2.batch = arg10
        args2.depth = arg12
        args2.dropout = arg11
        args2.lr = arg15
        args2.neurons1 = arg13
        args2.neurons2 = arg14

    print("Start training.")    

    list_number = []
    list_phe = []
    dictSeq = {}  
    dictSeq_1 = {}  

    scaler = StandardScaler()   

    df_data_raw = pd.read_csv(args2.env_path)

    env_factor = df_data_raw.shape[1]
    env_days = df_data_raw.shape[0]
    args2.enc_in = env_factor-2
    args2.c_out = env_factor-2

    env_code = list(OrderedDict.fromkeys(df_data_raw['env']))
    env_num = len(env_code)
    env_days = int(env_days/len(env_code))
    #print("env_num",env_num,"env_days",env_days)

    data_E_list = [] 
    for i in range(env_num):
        df_data = df_data_raw.iloc[i*env_days:(i+1)*env_days, 1:env_factor]
        time_data = df_data.iloc[:,0].reset_index(drop=True)
        env_data = df_data.iloc[:,1:].reset_index(drop=True)
        env_data = scaler.fit_transform(env_data)   
        env_data = pd.DataFrame(env_data)
        data_E = pd.concat([time_data, env_data],axis = 1)
        data_E_list.append(data_E)



    with open(args2.geno_path) as file: 
        reader = csv.reader(file)
        first_row = next(reader)  
        snp_len = len(first_row)-1
        #print("snp_len",snp_len)

        
        
    for i in range(env_num):
        with open(args2.geno_path) as file:
            num = '' 
            for line in islice(file, 1, None):
                num = line.split(",")[0]+str('_'+env_code[i])
                list_str = line.split(",")[1:snp_len+1] 
                list_int = [int(x) for x in list_str]
                dictSeq[num] = list_int
                dictSeq_1[num] = data_E_list[i]


    df = pd.read_csv(args2.pheno_path)
    filtered_columns = [col for col in df.columns if col.startswith(args2.pheno_name+"_")]
    filtered_columns = np.insert(filtered_columns, 0, "ID", axis=0)
    filtered_df = df.loc[:, filtered_columns]


    for i in range(env_num):
        list_str = []
        for index, row in filtered_df.iterrows():
            if row[0] + '_' + env_code[i] in dictSeq.keys():
                list_number.append(row[0] + '_' + env_code[i])
                item_phe = row[i+1]
                item_phe = float(item_phe)
                list_phe.append(item_phe)
        
    dt = open(args2.CVF_path, 'r')
    df = pd.read_csv(dt)

    best_accz = 0.0
    p_valuez = 0.0

    for val_num in range(1, env_num+1): 
        '''print("batch",args2.batch,
              "depth",args2.depth,
              "dropout",args2.dropout,
              "lr",args2.lr,
              "neurons1",args2.neurons1,
              "neurons2",args2.neurons2)'''
        best_acc = 0.0
        p_value = 0.0
        best_acc_0 = 0.0
        p_value_0 = 0.0
        list_number2 = []
        list_phe2 = []
        dictSeq2 = {}
        dictSeq2_1 = {}
        
        list_number3 = []
        list_phe3 = []
        dictSeq3 = {}
        dictSeq3_1 = {}
        
        val_data = df[df['CV']==val_num].index
        train_data = df[(df['CV']!=val_num)].index
        

        for h2 in range(int(len(train_data))): 
            list_number2.append(list_number[train_data[h2]])
            list_phe2.append(list_phe[train_data[h2]])
            dictSeq2[list_number2[h2]] = dictSeq[list_number2[h2]]
            dictSeq2_1[list_number2[h2]] = dictSeq_1[list_number2[h2]]
            
        for h3 in range(int(len(val_data))): 
            list_number3.append(list_number[val_data[h3]]) 
            list_phe3.append(list_phe[val_data[h3]])
            dictSeq3[list_number3[h3]] = dictSeq[list_number3[h3]]
            dictSeq3_1[list_number3[h3]] = dictSeq_1[list_number3[h3]]
            
        mdata_train = myDataset(list_number2, list_phe2, dictSeq2, dictSeq2_1) 
        hh = int(len(mdata_train))  
        
        mdata_val = myDataset(list_number3, list_phe3, dictSeq3, dictSeq3_1)
        hh1 = int(len(mdata_val)) 
        
        
        #print('*' * 25, 'di', "valid_num=",val_num, 'ci', '*' * 25)
        
        train_loader = DataLoader(
            dataset=mdata_train,
            batch_size=args2.batch,
            shuffle=True
        )
        
        val_loader = DataLoader(
            dataset=mdata_val,
            batch_size=args2.batch,
            shuffle=True,
        )
        

        net = GEFormer(args2, snp_len, env_days).to(args2.device)
        optimizer = optim.Adam(net.parameters(), args2.lr)    
        loss_func = nn.MSELoss()
        for epoch in range(100): 
            net.train()
            for j,(id, phe, dictseq, data_x, data_stamp) in enumerate(train_loader, 0):
                
                dictseq = torch.stack(dictseq).to(args2.device)
                phe = phe.clone().detach().to(args2.device) 
                phe = phe.float()
                dictseq = dictseq.float()
                data_x = data_x.to(args2.device)
                data_stamp = data_stamp.to(args2.device)

                optimizer.zero_grad()

                pred = net(dictseq, data_x, data_stamp).flatten()

                loss = loss_func(pred, phe)

                loss.backward()
                optimizer.step()

            net.eval()
            
            all_val_pred = []
            all_val_phe = []
            for jj, (id, phe, dictseq, data_x, data_stamp) in enumerate(val_loader, 0):
                dictseq = torch.stack(dictseq).to(args2.device)
                phe = phe.clone().detach().to(args2.device)  
                phe = phe.float()
                dictseq = dictseq.float()
                data_x = data_x.to(args2.device)
                data_stamp = data_stamp.to(args2.device)

                pred = net(dictseq, data_x, data_stamp).flatten()
                
                phe = phe.cpu().detach().numpy().tolist()
                all_val_phe.extend(phe)
                pred = pred.flatten().cpu().detach().numpy().tolist()
                all_val_pred.extend(pred)
                
            all_val_pred = np.asarray(all_val_pred)
            all_val_phe = np.asarray(all_val_phe)
            pccs = pearsonr(all_val_pred, all_val_phe)
            if pccs[0] > best_acc:
                best_acc = pccs[0]
                p_value = pccs[1]
                best_model = copy.deepcopy(net)
                
            elif pccs[0] < best_acc_0:
                best_acc_0 = pccs[0]
                p_value_0 = pccs[1]
                best_model_0 = copy.deepcopy(net)
        
        if best_acc == 0.0:
            pearson = (best_acc_0,p_value_0)
            torch.save(best_model_0, args2.model_path+"/model_"+str(val_num)+".pkl")
        else:
            pearson = (best_acc,p_value)
            torch.save(best_model, args2.model_path+"/model_"+str(val_num)+".pkl")
        print("Pearson = ({:.2f}, {:.4f})".format(pearson[0], pearson[1]))

        if best_accz < best_acc:
            best_accz = best_acc
            torch.save(best_model, args2.model_path+"/bestmodel.pkl")
    #print("best_accz = {:.2f}".format(best_accz)) 
    print("Training finished.")  
