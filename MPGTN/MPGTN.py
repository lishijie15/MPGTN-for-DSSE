import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.fft
import time
from layers.Embed_test import DataEmbedding, SpaceEmbedding
from layers.Conv_Blocks_dsc_tky import MobileNetV1
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    z, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def spilt(x, f):
    if f == 0:
        cs = torch.split(x, 1, dim=1)
        x = torch.concat(cs, dim=0).squeeze(1)
    else:
        cs = torch.split(x, 1, dim=2)
        x = torch.concat(cs, dim=0).squeeze(2)
    return x

def concat(b, x, f):
    if f == 0:
        x = torch.split(x, b, dim=0)
        x = torch.stack(x, dim=1)
    else:
        x = torch.split(x, b, dim=0)
        x = torch.stack(x, dim=2)
    return x

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class Times(nn.Module):
    def __init__(self, seq_len, out_len, top_k, d_model, d_ff, num_kernels):
        super(Times, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.k = top_k
        self.d_model = d_model
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(self.d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, self.d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()  #B: batch_size T: seq_len+pred_len N: d_model
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.out_len) % period != 0:
                length = (
                    ((self.seq_len + self.out_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.out_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.out_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.out_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class GCTblock_enc(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, d_model, seq_len, out_len, b_layers, top_k, d_ff, num_kernels):
        super(GCTblock_enc, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.k = top_k
        self.num_nodes = num_nodes
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.cheb_k = cheb_k
        self.d_ff =d_ff
        self.num_kernels = num_kernels
        self.b_layers = b_layers
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.5)
        self.AGCNs = nn.ModuleList([AGCN(self.d_model, self.d_model, self.cheb_k)
                                    for _ in range(self.b_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.Times = nn.ModuleList([Times(self.seq_len, self.out_len, self.k, self.d_model, self.d_ff, self.num_kernels)
                                    for _ in range(self.b_layers)])
        self.data_embedding = DataEmbedding(self.input_dim, self.d_model)
        self.spa_embedding = SpaceEmbedding(self.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.out_len + self.seq_len)
        self.projection = nn.Linear(self.d_model, self.output_dim, bias=True)
        self.node_linear = nn.Linear(self.num_nodes, self.d_model)
        # parameter-efficient design

    def forward(self, x, supports):
        # shape of x: (B, T, N, D)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        #embedding
        bat = x.shape[0]
        current_inputs = self.data_embedding(x)  # [B,T,N,D]
        current_inputs = self.predict_linear(current_inputs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_res = self.predict_linear(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        for i in range(self.b_layers):
            current_inputs = spilt(current_inputs, 0)
            state = self.AGCNs[i](current_inputs, supports)
            state = concat(bat, state, 0)
            t_enc_out = spilt(state, 1)
            t_enc_out = self.layer_norm(self.Times[i](t_enc_out))
            current_inputs = concat(bat, t_enc_out, 1)
        return current_inputs, x_res

class GCTblock_dec(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, seq_len, out_len, d_model, b_layers, top_k, d_ff, num_kernels):
        super(GCTblock_dec, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.k = top_k
        self.num_nodes = num_nodes
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.d_ff =d_ff
        self.cheb_k = cheb_k
        self.num_kernels = num_kernels
        self.b_layers = b_layers
        self.d_model = 2*d_model
        self.AGCNs = nn.ModuleList([AGCN(self.input_dim, self.output_dim, self.cheb_k)
                                    for _ in range(self.b_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.Times_de = nn.ModuleList([Times(self.seq_len, self.out_len, self.k, self.d_model, self.d_ff, self.num_kernels)
                                    for _ in range(self.b_layers)])
        self.enc_embedding = DataEmbedding(self.input_dim, self.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.out_len + self.seq_len)
        self.projection = nn.Linear(self.num_nodes, self.output_dim, bias=True)
        # parameter-efficient design

    def forward(self, h_b, supports):
        current_inputs = h_b
        assert h_b.shape[2] == self.num_nodes and h_b.shape[3] == self.input_dim
        bat = h_b.shape[0]
        for i in range(self.b_layers):
            current_inputs = spilt(current_inputs, 0)
            state = self.AGCNs[i](current_inputs, supports)
            state = concat(bat, state, 0)
            t_dec_out = spilt(state, 1)
            # TimesNet
            t_dec_out = self.layer_norm(self.Times_de[i](t_dec_out))
            current_inputs = concat(bat, t_dec_out, 1)
        return current_inputs



class MPGTN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, seq_len, out_len, b_layers, d_model, top_k, d_ff, num_kernels,
                 cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000):
        super(MPGTN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.b_layers = b_layers
        self.d_model = d_model
        self.top_k = top_k
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.layer_norm = nn.LayerNorm(self.num_nodes)
        self.output_dim = output_dim
        self.out_len = out_len
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.weights = nn.Parameter(torch.FloatTensor(self.input_dim, 32)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(32))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.GCT_enc = GCTblock_enc(self.num_nodes, self.input_dim, self.output_dim, self.cheb_k,
                                    self.d_model, self.seq_len, self.out_len, self.b_layers, self.top_k, self.d_ff, self.num_kernels)

        # decoder
        self.decoder_dim = self.d_model + self.mem_dim
        self.GCT_dec = GCTblock_dec(self.num_nodes, self.d_model + self.mem_dim, self.decoder_dim, self.cheb_k,
                                    self.seq_len, self.out_len, self.d_model, self.b_layers, self.top_k, self.d_ff, self.num_kernels)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        #TimesNet
        self.convTran = nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=(1, 32), stride=(1, 1))
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.d_model, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        return value, query, pos, neg

            
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        # mem_num：number of prototypes
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  #E,ET
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        # supports = [0.2*adjs[0]*g1, 0.2*adjs[1]*g2]
        supports = [g1, g2]
        bat = x.shape[0]

        hm_t, x_res = self.GCT_enc(x, supports)
        hm_t = hm_t + x_res

        #meta bank
        m_t = spilt(hm_t, 0)
        h_att, query, pos, neg = self.query_memory(m_t)
        h_t = torch.cat([m_t, h_att], dim=-1)
        h_b = concat(bat, h_t, 0)

        # ht_list = [h_b]*self.e_layers
        h_de = self.GCT_dec(h_b, supports)
        h_de = h_de + h_b + x_res
        go = self.proj(h_de)
        output = go[:, -self.out_len:, :, :]

        return output, h_att, query, pos, neg
