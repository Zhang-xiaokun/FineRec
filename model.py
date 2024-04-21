import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
import pickle
import torch.nn.utils.rnn as rnn_utils



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class Opin_dim_trans(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(Opin_dim_trans, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, int(input_size/8))
        self.mlp_2 = nn.Linear(int(input_size/8), out_size)
        # self.mlp_3 = nn.Linear(input_size, input_size)
        # self.mlp_4 = nn.Linear(input_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_3(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_4(emb_trans)))
        return emb_trans

def opin_ui_emb(q_layer, k_layer, v_layer, emb_q, emb_k, matrix, opin_emb_table, emb_size):

    q_emb = q_layer(emb_q)
    k_emb = k_layer(emb_k)
    # emb_v = v_layer(emb_v)

    q_expand = q_emb.detach().expand_as(k_emb)
    w_agg = torch.sum(q_expand.detach() * k_emb.detach(), 1)
    # w_agg = nn.Softmax(dim=-1)(w_agg)
    w_expand = w_agg.detach().unsqueeze(1).repeat(1, emb_size)

    # chunk
    results = trans_to_cuda(torch.Tensor())
    for chunk_t in torch.chunk(matrix, 10, dim=0):

        emb_v = v_layer(opin_emb_table[chunk_t])
        w_expand_3 = w_expand.detach().unsqueeze(0).expand_as(emb_v)
        opin_out = w_expand_3.detach() * emb_v.detach()
        del w_expand_3
        del emb_v

        temp_re = torch.sum(opin_out.detach(), 1)
        results = torch.cat([results, temp_re], 0)

    # print("Memory Allocated: ", torch.cuda.memory_allocated() / (1024 ** 3), "GB")
    return results

class UiMatrix(Module):
    def __init__(self, dataset, ui_layers, graph_layers, whole_layers, chunk_embSize, n_user, n_item, attr_num, n_opi, num_heads):
        super(UiMatrix, self).__init__()
        self.emb_size = chunk_embSize
        self.ui_layers = ui_layers
        self.graph_layers = graph_layers
        self.whole_layers = whole_layers
        self.n_user = n_user
        self.n_item = n_item
        self.attr_num = attr_num
        self.n_opi = n_opi
        self.num_heads = num_heads

        # introducing opinion embeddings
        opin_path = './datasets/' + dataset + '/opinMatrixpca' + str(self.emb_size) + '.npy'
        opinWeights = np.load(opin_path)
        self.opi_emb = nn.Embedding(self.n_opi, self.emb_size)
        opinWeights_np = np.array(opinWeights)
        self.opi_emb.weight.data.copy_(torch.from_numpy(opinWeights_np))

        # self.opi_emb = nn.Embedding(self.n_opi, self.emb_size)

        self.q_layer = torch.nn.ModuleList([MLP_one_layer(self.emb_size, self.emb_size, dropout=0.5) for _ in
                                            range(attr_num*2)])
        self.k_layer = torch.nn.ModuleList([MLP_one_layer(self.emb_size, self.emb_size, dropout=0.5) for _ in
                                            range(attr_num*2)])
        self.v_layer = torch.nn.ModuleList([MLP_one_layer(self.emb_size, self.emb_size, dropout=0.5) for _ in
                                            range(attr_num*2)])

        self.attr_graph_layer = torch.nn.ModuleList([MLP_one_layer(self.emb_size, self.emb_size, dropout=0.5) for _ in
                                            range(attr_num * 2)])
        self.MSA_user = MultiHeadSelfAttention(self.emb_size, self.num_heads)
        self.MSA_item = MultiHeadSelfAttention(self.emb_size, self.num_heads)

        self.dim_tran = Opin_dim_trans(768, self.emb_size)

        self.emb_drop = nn.Dropout(0.1)
        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def opin_update_user(self,item_table, opin_emb_table, attr_embedding, user_io):
        item_seqs = trans_to_cuda(torch.LongTensor(user_io[0]))
        opin_seqs = trans_to_cuda(torch.LongTensor(user_io[1]))
        emb_k = trans_to_cuda(item_table[item_seqs])
        emb_v = trans_to_cuda(opin_emb_table[opin_seqs])
        q_emb = self.q_u_layer(attr_embedding)
        k_emb = self.k_u_layer(emb_k)

        q_expand = q_emb.detach().expand_as(k_emb)
        w_agg = torch.sum(q_expand.detach() * k_emb.detach(), -1)
        mask = torch.where(item_seqs > 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        w_agg = mask * w_agg
        # w_agg = nn.Softmax(dim=-1)(w_agg)
        w_expand = w_agg.detach().unsqueeze(2).repeat(1,1,self.emb_size)
        w_expand = trans_to_cuda(w_expand)
        emb_v = self.v_u_layer(emb_v)
        opin_out = w_expand.detach() * emb_v.detach()
        user_embedding = torch.sum(opin_out.detach(), 1)
        return user_embedding

    def opin_update_ui(self,q_layer, k_layer, v_layer, item_table, opin_emb_table, attr_embedding, user_emb, user_io):
        item_seqs = trans_to_cuda(torch.LongTensor(user_io[0]))
        opin_seqs = trans_to_cuda(torch.LongTensor(user_io[1]))

        item_emb = trans_to_cuda(item_table[item_seqs])
        opin_emb = trans_to_cuda(opin_emb_table[opin_seqs])
        user_emb = trans_to_cuda(user_emb)
        attr_embedding= trans_to_cuda(attr_embedding)

        attr_expand = attr_embedding.expand_as(opin_emb)

        k_emb = self.dropout50(attr_expand + item_emb)
        # k_emb = self.dropout30(torch.cat([item_emb * opin_emb, item_emb + opin_emb], -1))
        v_emb = item_emb +  opin_emb
        k_emb = k_layer(k_emb)
        v_emb = v_layer(v_emb)


        # q_expand = self.dropout30(torch.cat([user_emb, attr_expand], -1))
        q_expand = q_layer(user_emb)
        q_emb = q_expand.unsqueeze(1).expand_as(k_emb)
        w_agg = torch.sum(q_emb.detach() * k_emb.detach(), -1)
        mask = torch.where(item_seqs > 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        w_agg = mask * w_agg
        # w_agg = nn.Softmax(dim=-1)(w_agg)
        w_expand = w_agg.detach().unsqueeze(2).repeat(1,1,self.emb_size)
        w_expand = trans_to_cuda(w_expand)
        opin_out = w_expand.detach() * v_emb.detach()
        user_embedding = torch.sum(opin_out.detach(), 1)
        return user_embedding

    def opin_update_item(self,user_table, opin_emb_table, attr_embedding, item_uo):
        item_seqs = trans_to_cuda(torch.LongTensor(item_uo[0]))
        opin_seqs = trans_to_cuda(torch.LongTensor(item_uo[1]))
        emb_k = trans_to_cuda(user_table[item_seqs])
        emb_v = trans_to_cuda(opin_emb_table[opin_seqs])
        q_emb = self.q_i_layer(attr_embedding)
        k_emb = self.k_i_layer(emb_k)

        q_expand = q_emb.detach().expand_as(k_emb)
        w_agg = torch.sum(q_expand.detach() * k_emb.detach(), -1)
        mask = torch.where(item_seqs > 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        w_agg = mask * w_agg
        # w_agg = nn.Softmax(dim=-1)(w_agg)
        w_expand = w_agg.detach().unsqueeze(2).repeat(1,1,self.emb_size)
        w_expand = trans_to_cuda(w_expand)
        emb_v = self.v_i_layer(emb_v)
        opin_out = w_expand.detach() * emb_v.detach()
        item_embedding = torch.sum(opin_out.detach(), 1)
        return item_embedding

    def forward(self, user_io, item_io, attr_embs, user_embs, item_embs):
        # attr_embedding = self.emb_drop(attr_embs)
        # zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # attr_emb_table = torch.cat([zeros, attr_embs], 0)
        user_chunk_embedding = []
        item_chunk_embedding = []
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        opin_embedding = self.opi_emb.weight
        # opin_embedding = self.dim_tran(opin_embedding)
        opin_emb_table = torch.cat([zeros, opin_embedding], 0)
        opin_emb_table = trans_to_cuda(opin_emb_table)
        for j in range(self.attr_num):
            user_chunk_embedding.append(trans_to_cuda(user_embs[j].weight))
            item_chunk_embedding.append(trans_to_cuda(item_embs[j].weight))
        # opinion updating user&item embeddings
        for i in range(self.ui_layers):
            for j in range(self.attr_num):
                attr_embedding = attr_embs(torch.cuda.LongTensor([j]))
                user_table = torch.cat([zeros, user_chunk_embedding[j]], 0)

                item_table = torch.cat([zeros, item_chunk_embedding[j]], 0)

                user_chunk_embedding[j] = user_chunk_embedding[j] + self.opin_update_ui(self.q_layer[j], self.k_layer[j], self.v_layer[j], item_table, opin_emb_table, attr_embedding, user_chunk_embedding[j], user_io[j])
                item_chunk_embedding[j] = item_chunk_embedding[j] + self.opin_update_ui(self.q_layer[j+self.attr_num], self.k_layer[j+self.attr_num], self.v_layer[j+self.attr_num], user_table, opin_emb_table, attr_embedding, item_chunk_embedding[j], item_io[j])
                # if j == 0:
                #     user_sa = user_chunk_embedding[j].unsqueeze(1)
                #     item_sa = item_chunk_embedding[j].unsqueeze(1)
                # else:
                #     user_sa = torch.cat((user_sa, user_chunk_embedding[j].unsqueeze(1)), 1)
                #     item_sa = torch.cat((item_sa, item_chunk_embedding[j].unsqueeze(1)), 1)

            # user_whole = self.MSA_user(user_sa)
            # item_whole = self.MSA_item(item_sa)
            # for j in range(self.attr_num):
            #     user_chunk_embedding[j] = user_whole[:, j, :]
            #     item_chunk_embedding[j] = item_whole[:, j, :]

        # user-item interaction updating user&item embedding
        # for i in range(self.graph_layers):
        #     for j in range(self.attr_num):
        #         # attr_embedding = attr_embs(torch.cuda.LongTensor([j]))
        #         # u-i matrix weights = 1
        #         values = np.ones_like(adjacency[j].data)
        #         indices = np.vstack((adjacency[j].row, adjacency[j].col))
        #         i = torch.LongTensor(indices)
        #         v = torch.LongTensor(values)
        #         # n_u * n_i
        #         matrix = torch.sparse.FloatTensor(i, v, torch.Size(adjacency[j].shape))
        #         # item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embedding)
        #         matrix = trans_to_cuda(matrix).to_dense()
        #         matrix_T = torch.transpose(matrix, 1, 0)
        #
        #
        #
        #         user_chunk_embedding[j] = user_chunk_embedding[j] + self.graph_aggr(self.attr_graph_layer[j], user_chunk_embedding[j], item_chunk_embedding[j], matrix)
        #         item_chunk_embedding[j] = item_chunk_embedding[j] + self.graph_aggr(self.attr_graph_layer[j+self.attr_num], item_chunk_embedding[j], user_chunk_embedding[j], matrix_T)

        # output user/item chunks embeddings
        return user_chunk_embedding, item_chunk_embedding
    # def opin_ui_chunk(self, emb_q, emb_k, emb_v):
    #     q_emb = self.q_layer(emb_q)
    #     k_emb = self.k_layer(emb_k)
    #     v_emb = self.v_layer(emb_v)
    #     q_expand = q_emb.expand_as(k_emb)
    #     w_agg = torch.sum(q_expand*k_emb, 1)
    #     # w_agg = nn.Softmax(dim=-1)(w_agg)
    #     w_expand = w_agg.unsqueeze(1).repeat(1,self.emb_size)
    #     w_expand_3 = w_expand.unsqueeze(0).expand_as(v_emb)
    #
    #     opin_out = w_expand_3*v_emb
    #     opin_out = torch.sum(opin_out, 1)
    #
    #     return opin_out

    def graph_aggr(self, weight, m1, m2, mat):
        w_layer = weight(m2)
        temp1 = torch.matmul(m1, w_layer.transpose(-1, -2))
        temp2 = temp1*mat
        re_mat = m1 + torch.matmul(temp2, m2)
        return re_mat
# class LineConv(Module):
#     def __init__(self, layers,batch_size,emb_size=100):
#         super(LineConv, self).__init__()
#         self.emb_size = emb_size
#         self.batch_size = batch_size
#         self.layers = layers
#     def forward(self, item_embedding, D, A, session_item, session_len):
#         zeros = torch.cuda.FloatTensor(1,self.emb_size).fill_(0)
#         # zeros = torch.zeros([1,self.emb_size])
#         item_embedding = torch.cat([zeros, item_embedding], 0)
#         seq_h = []
#         for i in torch.arange(len(session_item)):
#             seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
#         seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
#         session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
#         session = [session_emb_lgcn]
#         DA = torch.mm(D, A).float()
#         for i in range(self.layers):
#             session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
#             session.append(session_emb_lgcn)
#         session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
#         session_emb_lgcn = torch.sum(session1, 0)
#         return session_emb_lgcn

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num=2, dropout=0.1, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size

        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        #         print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None):
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to('cuda')
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)

        # head_num * batch_size * session_length * head_dim
        outputs = weights.matmul(values)
        # batch_size * session_length * hidden_size
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.relu(self.final1(x))
        x = self.final2(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=4, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs



class MLP_one_layer(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_one_layer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s = nn.Linear(input_size, out_size)

    def forward(self, emb_trans):
        results = self.dropout(self.activate(self.mlp_s(emb_trans)))
        return results

class MLP_sim(torch.nn.Module):
    def __init__(self, input_size=64, dropout=0.2):
        super(MLP_sim, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s1 = nn.Linear(input_size, int(input_size / 2))
        self.mlp_s2 = nn.Linear(int(input_size / 2), int(input_size / 8))
        self.mlp_s3 = nn.Linear(int(input_size / 8), int(input_size / 16))
        self.mlp_s4 = nn.Linear(int(input_size / 16), 1)

    def forward(self, emb):
        results = self.dropout(self.activate(self.mlp_s1(emb)))
        results = self.dropout(self.activate(self.mlp_s2(results)))
        results = self.dropout(self.activate(self.mlp_s3(results)))
        results = self.dropout(self.activate(self.mlp_s4(results)))
        return results

class MLP_meger(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(MLP_meger, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s1 = nn.Linear(input_size, int(input_size / 4))
        # self.mlp_s2 = nn.Linear(int(input_size / 4), int(input_size / 8))
        self.mlp_s3 = nn.Linear(int(input_size / 4), output_size)

    def forward(self, emb):
        results = self.dropout(self.activate(self.mlp_s1(emb)))
        # results = self.dropout(self.activate(self.mlp_s2(results)))
        results = self.dropout(self.activate(self.mlp_s3(results)))
        return results

class MLP_seq(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(MLP_seq, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s1 = nn.Linear(input_size, int(input_size / 2))
        self.mlp_s2 = nn.Linear(int(input_size / 2), output_size)

    def forward(self, emb):
        results = self.dropout(self.activate(self.mlp_s1(emb)))
        results = self.dropout(self.activate(self.mlp_s2(results)))
        return results
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FineRec(Module):
    def __init__(self, user_adj, item_adj, ui_list, iu_list, user_io, item_uo, n_attr, n_opi, n_user, n_item, lr, ui_layer, graph_layer, whole_layer, attr_num, l2, dataset, num_heads=4, emb_size=100, chunk_embSize=16, batch_size=100):
        super(FineRec, self).__init__()
        self.emb_size = emb_size
        self.chunk_embSize = chunk_embSize
        self.n_attr = n_attr
        self.n_opi = n_opi
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_user = n_user
        self.n_item = n_item
        self.attr_num = attr_num
        self.ui_layer = ui_layer
        self.graph_layer = graph_layer
        self.whole_layer = whole_layer
        self.L2 = l2
        self.lr = lr
        self.num_heads = num_heads

        # self.transformers = torch.nn.ModuleList([TransformerLayer(emb_size, head_num=num_heads, dropout=0.6,
        #                                                           attention_dropout=0,
        #                                                           initializer_range=0.02) for _ in
        #                                          range(layers)])

        self.user_chunk_embs = torch.nn.ModuleList([nn.Embedding(self.n_user, self.chunk_embSize) for _ in
                                                 range(attr_num)])
        self.item_chunk_embs = torch.nn.ModuleList([nn.Embedding(self.n_item, self.chunk_embSize) for _ in
                                                    range(attr_num)])

        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)

        self.user_adj = user_adj
        self.item_adj = item_adj
        self.ui_list = ui_list
        self.iu_list = iu_list
        self.user_io = user_io
        self.item_uo = item_uo
        self.UiMatrix = UiMatrix(self.dataset, self.ui_layer, self.graph_layer, self.whole_layer, self.chunk_embSize, self.n_user, self.n_item, self.attr_num, self.n_opi, self.num_heads)
        self.merge_uc_mlp = MLP_meger(chunk_embSize*attr_num, emb_size)
        self.merge_ic_mlp = MLP_meger(chunk_embSize * attr_num, emb_size)
        self.inter_u_m = MLP_seq(chunk_embSize*attr_num, chunk_embSize*attr_num)
        self.inter_i_m = MLP_seq(chunk_embSize*attr_num, chunk_embSize*attr_num)

        self.merge_sl = MLP_seq(2*emb_size, emb_size)

        # introducing attribute&opinion embeddings
        # opin_path = './datasets/' + dataset + '/opinMatrixpca' + str(chunk_embSize) + '.npy'
        # opinWeights = np.load(opin_path)
        # self.opi_emb = nn.Embedding(self.n_opi, self.chunk_embSize)
        # opinWeights_np = np.array(opinWeights)
        # self.opi_emb.weight.data.copy_(torch.from_numpy(opinWeights_np))

        self.attr_emb = nn.Embedding(self.attr_num, self.chunk_embSize)

        self.query_seq = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.key_seq = nn.Linear(self.emb_size, self.emb_size)
        self.value_seq = nn.Linear(self.emb_size, self.emb_size)

        self.gate_w1 = nn.Linear(self.emb_size, self.emb_size)
        self.gate_w2 = nn.Linear(self.emb_size, self.emb_size)

        # self.pos_embedding = nn.Embedding(2000, self.emb_size)


        self.active = nn.ReLU()
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()

        # 添加权重矩阵，学习权重

        # # self_attention
        # if self.emb_size % num_heads != 0:  # 整除
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (emb_size, num_heads))
        #     # 参数定义
        # # self.num_heads = num_heads  # 4
        # self.attention_head_size = int(emb_size / num_heads)  # 16  每个注意力头的维度
        # self.all_head_size = int(self.num_heads * self.attention_head_size)
        # # query, key, value 的线性变换（上述公式2）
        # self.query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        # self.key = nn.Linear(self.emb_size, self.emb_size)
        # self.value = nn.Linear(self.emb_size, self.emb_size)

        # self.query_id = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        # self.key_id = nn.Linear(self.emb_size, self.emb_size)
        # self.value_id = nn.Linear(self.emb_size, self.emb_size)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, user_embs, item_embs, u_id, session_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)

        user_emb_table = torch.cat([zeros, user_embs], 0)
        item_emb_table = torch.cat([zeros, item_embs], 0)

        # id seq emb
        user_emb = user_emb_table[u_id]
        seq_item_emb = item_emb_table[session_item]
        # mask = mask.float().unsqueeze(-1)
        # Self-attention to get session emb
        # attention_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [bs, 1, 1, seqlen] 增加维度
        # attention_mask = (1.0 - attention_mask) * -10000.0
        #
        # mixed_query_layer = self.query_seq(seq_item_emb)  # [bs, seqlen, hid_size]
        # mixed_key_layer = self.key_seq(seq_item_emb)  # [bs, seqlen, hid_size]
        # mixed_value_layer = self.value_seq(seq_item_emb)  # [bs, seqlen, hid_size]
        #
        # attention_head_size = int(self.emb_size / self.num_heads)
        # query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        # value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        # attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        # attention_scores = attention_scores + attention_mask
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        #
        # # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        # context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        # sa_result = context_layer.view(*new_context_layer_shape)
        # # last hidden state as price preferences
        # item_pos = torch.tensor(range(1, seq_item_emb.size()[1] + 1), device='cuda')
        # item_pos = item_pos.unsqueeze(0).expand_as(session_item)
        #
        # item_pos = item_pos * mask.squeeze(2)
        # item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        # last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
        #                          torch.tensor([0.0], device='cuda'))
        # as_last_unit = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        # seq_emb = torch.sum(as_last_unit, 1)
        #
        # gate = torch.tanh(self.gate_w1(seq_emb) + self.gate_w2(user_emb))

        # resluts = user_emb + gate * seq_emb
        # seq_emb = self.merge_sl(torch.cat([user_emb, last_emb], 1))
        # seq_emb = 0
        len_seq = trans_to_cuda(torch.sum(mask, -1)).unsqueeze(1)
        seq_emb = torch.div(torch.sum(seq_item_emb, 1), len_seq)
        resluts = user_emb + seq_emb

        return resluts

    def id_co_loss(self, item_emb, pos_item, neg_item, weights):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_emb_table = torch.cat([zeros, item_emb], 0)
        item_batch = torch.chunk(item_emb, self.n_node//512, 0)
        pos_batch = torch.chunk(torch.tensor(pos_item), self.n_node // 512, 0)
        neg_batch = torch.chunk(torch.tensor(neg_item), self.n_node // 512, 0)
        weight_batch = torch.chunk(torch.tensor(weights), self.n_node // 512, 0)
        con_loss = 0
        tau = 1
        for id_temp_emb, pos_temp, neg_temp, weight_temp in zip(item_batch, pos_batch, neg_batch, weight_batch):
            pos_temp_emb = item_emb_table[pos_temp]
            neg_temp_emb = item_emb_table[neg_temp]
            weight_temp = -trans_to_cuda(weight_temp)
            weight_temp = weight_temp / (torch.sum(weight_temp, 1).unsqueeze(1).repeat(1,weight_temp.shape[1]) + 1e-8)
            pos_temp_emb = pos_temp_emb * weight_temp.unsqueeze(2).repeat(1,1,pos_temp_emb.shape[2])
            pos_mean_emb = torch.sum(pos_temp_emb, 1)
            pos_dis = self.cos_sim(id_temp_emb, pos_mean_emb)
            # pos_dis = self.sim_cal_w(id_temp_emb, pos_mean_emb, self.sim_w_i1)
            fenzi = pos_dis/tau
            # fenzi = torch.log10(torch.exp(fenzi, out=None))
            fenzi = torch.exp(fenzi)
            id_temp_emb_expand = id_temp_emb.unsqueeze(1).repeat(1, neg_temp_emb.shape[1], 1)
            neg_dis = torch.exp(self.cos_sim(id_temp_emb_expand, neg_temp_emb)/tau)
            # neg_dis = self.sim_cal_w(id_temp_emb_expand, neg_temp_emb, self.sim_w_i2)
            # neg_dis = torch.exp(neg_dis / tau)
            fenmu = torch.sum(neg_dis, 1)
            fenmu = fenzi + fenmu
            fenzi = torch.log10(fenzi)
            fenmu = torch.log10(fenmu)
            temp_loss = fenmu - fenzi
            temp_loss = torch.sum(temp_loss, 0)
            con_loss += temp_loss
            # neg_embedding = torch.sum(neg_temp_emb, 1)


        return con_loss

    def seq_pro_loss(self, sess_id, sess_id_pro, sess_mo, sess_mo_pro):
        s_id_loss1 = self.cos_sim(self.mlp_seq1(sess_id), self.mlp_seq2(sess_id_pro))
        s_id_loss2 = self.cos_sim(self.mlp_seq3(sess_id), self.mlp_seq4(sess_mo))
        # s_id_loss1 = self.sim_cal_w(sess_id, sess_id_pro, self.sim_w_p1u)
        # s_id_loss2 = self.sim_cal_w(sess_id, sess_mo, self.sim_w_p1d)


        s_id_loss = torch.log10(torch.exp(s_id_loss1)) - torch.log10(torch.exp(s_id_loss2)+torch.exp(s_id_loss1))
        # s_id_loss = self.relu(s_id_loss)
        # s_id_loss = s_id_loss
        s_id_loss = torch.sum(s_id_loss, 0)
        s_mo_loss1 = self.cos_sim(self.mlp_seq4(sess_mo), self.mlp_seq5(sess_mo_pro))
        s_mo_loss2 = self.cos_sim(self.mlp_seq7(sess_mo), self.mlp_seq8(sess_id))
        # s_mo_loss1 = self.sim_cal_w(sess_mo, sess_mo_pro, self.sim_w_p2u)
        # s_mo_loss2 = self.sim_cal_w(sess_mo, sess_id, self.sim_w_p2d)

        s_mo_loss = torch.log10(torch.exp(s_mo_loss1)) - torch.log10(torch.exp(s_mo_loss2)+torch.exp(s_mo_loss1))
        # s_mo_loss = self.relu(s_mo_loss)
        # s_mo_loss = s_mo_loss
        s_mo_loss = torch.sum(s_mo_loss, 0)
        loss_pro = s_id_loss + s_mo_loss
        return -loss_pro

    def seq_cri_loss(self, item_emb, text_emb, sess_id_emb, sess_mo_emb, flag, lab):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_emb_table = torch.cat([zeros, item_emb], 0)
        text_emb_table = torch.cat([zeros, text_emb], 0)
        lab_id_emb = item_emb_table[lab]
        lab_mo_emb = text_emb_table[lab]
        id_dis = self.cos_sim(sess_id_emb, lab_id_emb)
        mo_dis = self.cos_sim(sess_mo_emb, lab_mo_emb)
        # id_dis = self.sim_cal_w(sess_id_emb, lab_id_emb, self.sim_w_c1u)
        # mo_dis = self.sim_cal_w(sess_mo_emb, lab_mo_emb, self.sim_w_c1d)

        cri_id_mo = (torch.log10(torch.exp(id_dis)) - torch.log10(torch.exp(mo_dis) + torch.exp(id_dis)))*flag
        cri_loss = torch.sum(cri_id_mo,0)
        return -cri_loss

    def sim_cal_w(self, emb1, emb2, mat_w):
        sim = emb1 * mat_w(emb2)
        sim = torch.sum(sim,-1)
        return sim

    def sim_cal_MLP(self, emb1, emb2, mat_w):
        sim = torch.cat([emb1, emb2], -1)
        sim = self.dropout3(sim)
        sim = mat_w(sim)
        return sim

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def agg_ui_whole(self, item_chunk, user_chunk):
        user_embs = user_chunk[0]
        item_embs = item_chunk[0]
        for i in range(1, self.attr_num):
            user_embs = torch.cat((user_embs, user_chunk[i]), 1)
            item_embs = torch.cat((item_embs, item_chunk[i]), 1)
        user_embs = trans_to_cuda(self.dropout(user_embs))
        item_embs = trans_to_cuda(self.dropout(item_embs))
        for i in range(0, self.whole_layer):
            item_embs_ui = self.inter_ui_meger(item_embs, user_embs, self.inter_i_m, self.iu_list)
            user_embs_ui = self.inter_ui_meger(user_embs, item_embs, self.inter_u_m, self.ui_list)
            item_embs_ii = self.get_embedding(self.item_adj, item_embs)
            user_embs_uu = self.get_embedding(self.user_adj, user_embs)
            user_embs =user_embs + (user_embs_ui + user_embs_uu) / 2
            item_embs =item_embs + (item_embs_ui + item_embs_ii) / 2
        return user_embs, item_embs

    def inter_ui_meger(self, user_emb, item_emb, w_layer, ui_seq):
        zeros = torch.cuda.FloatTensor(1, self.chunk_embSize*self.attr_num).fill_(0)
        item_table = torch.cat([zeros, item_emb], 0)
        item_emb_seq = item_table[ui_seq]
        user_expand = user_emb.unsqueeze(1).expand_as(item_emb_seq)
        item_emb_seq = w_layer(item_emb_seq)
        w_agg = torch.sum(user_expand.detach() * item_emb_seq.detach(), -1)
        mask = torch.where(trans_to_cuda(torch.tensor(ui_seq)) > 0, torch.tensor([1.0], device='cuda'),
                           torch.tensor([0.0], device='cuda'))
        w_agg = w_agg * mask
        w_expand = w_agg.detach().unsqueeze(2).repeat(1, 1, self.chunk_embSize*self.attr_num)
        w_expand = trans_to_cuda(w_expand)
        res_emb = w_expand * item_emb_seq
        res_emb = torch.sum(res_emb.detach(), 1)
        return res_emb
    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        mat_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embedding)
        mat_adj = trans_to_cuda(mat_adj)
        embeddings = torch.mm(mat_adj.to_dense(), trans_to_cuda(embedding))
        return embeddings

    def forward(self, u_id, pad_seq, mask):
        # session_item 是一个batch里的所有session [[23,34,0,0],[1,3,4,0]]
        # item_emb = trans_to_cuda(self.embedding.weight)
        # text_emb = trans_to_cuda(self.text_embedding.weight)
        attr_embs = trans_to_cuda(self.attr_emb)
        # embs[0], embs[1]
        user_chunks = trans_to_cuda(self.user_chunk_embs)
        item_chunks = trans_to_cuda(self.item_chunk_embs)
        # return user_chunk_embs, item_chunk_embs
        user_chunk_update, item_chunk_update = self.UiMatrix(self.user_io, self.item_uo, attr_embs, user_chunks, item_chunks)

        # item_chunk_update = []
        # user_chunk_update = []
        # for x in range(self.attr_num):
        #     item_chunk_update.append(item_chunks[x].weight)
        #     user_chunk_update.append(user_chunks[x].weight)


        user_embs, item_embs = self.agg_ui_whole(item_chunk_update, user_chunk_update)

        user_embs = self.merge_uc_mlp(user_embs)
        item_embs = self.merge_ic_mlp(item_embs)
        # obtain session emb
        seq_emb = self.generate_sess_emb(user_embs, item_embs, u_id, pad_seq, mask)  #  seq embeddings in batch



        return seq_emb, item_embs


def perform(model, i, data):
    user_id, pad_seq, mask, tar = data.get_slice(i) # 得到一个batch里的数据
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    user_id = trans_to_cuda(torch.Tensor(user_id).long())
    pad_seq = trans_to_cuda(torch.Tensor(pad_seq).long())
    user_emb, item_emb = model(user_id, pad_seq, mask)
    scores = torch.mm(user_emb, torch.transpose(item_emb, 1, 0))
    scores = trans_to_cuda(scores)
    return tar, scores

# def infer(model, i, data):
#     tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i) # 得到一个batch里的数据
#     # A_hat, D_hat = data.get_overlap(session_item)
#     session_item = trans_to_cuda(torch.Tensor(session_item).long())
#     session_len = trans_to_cuda(torch.Tensor(session_len).long())
#     # A_hat = trans_to_cuda(torch.Tensor(A_hat))
#     # D_hat = trans_to_cuda(torch.Tensor(D_hat))
#     tar = trans_to_cuda(torch.Tensor(tar).long())
#     flag = trans_to_cuda(torch.Tensor(flag).long())
#     mask = trans_to_cuda(torch.Tensor(mask).long())
#     reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
#     item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, con_loss = model(session_item, flag, session_len, reversed_sess_item, mask)
#     scores_co = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
#     scores_mo = torch.mm(sess_mo_emb, torch.transpose(item_mo_emb, 1, 0))
#     scores = scores_co + scores_mo
#     scores = trans_to_cuda(scores)
#     return tar, scores, con_loss

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) #将session随机打乱，每x个一组（#session/batch_size)
    for i in slices:
        model.zero_grad()
        targets, scores = perform(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores = perform(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


