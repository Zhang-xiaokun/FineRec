import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter
import random

import numpy as np
import torch
def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if reproducibility:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    # else:
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = False


def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #统计session中不同item，去重，并按照item_id排序
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:session长度累加和; indices:item_id 减1, 由每个session内item组成; data:item在session内的权重，全部为1.
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def data_easy_masks(mat, n_row, n_col):
    data, indices, indptr  = mat[0], mat[1], mat[2]

    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False,is_train=False, n_attr=None, n_user=None, n_item=None):
        # data formulation: 0:user_id; 1:x{t-1}; 2:label;
        # 3:item-co-matrix, 4:user-co-matrix, 5:user-item-list, 6:item-user-list
        # 7-13: user_attr1-7:item_seqs,opin_seqs,
        # 14-20:item_attr1-7:user_seqs, opin_seqs

        # 5-11: attr_coo_matrix -> user-item,
        # 12-18:user_attr1-7:item_seqs,opin_seqs,
        # 19-25:item_attr1-7:user_seqs, opin_seqs
        self.u_id = np.asarray(data[0])  # sessions, item_seq
        self.last_id = np.asarray(data[1])
        self.targets = np.asarray(data[2])
        if is_train:
            self.adjacency = []
            self.user_io = []
            self.item_uo = []
            self.item_adj = data_easy_masks(data[3], n_item, n_item).tocoo()
            self.user_adj = data_easy_masks(data[4], n_user, n_user).tocoo()
            self.ui_list = data[5]
            self.iu_list = data[6]
            # for x in range(5,5+n_attr):
            #     self.adjacency.append(data_easy_masks(data[x], n_user, n_item).tocoo())  # 10000 * 6558 #user * #items content:opinions 稀疏矩阵存储
            for x in range(7, 7+n_attr):
                self.user_io.append(data[x])
                self.item_uo.append(data[x+n_attr])
            # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
            # BH_T = BH_T.T
            # H = H_T.T
            # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
            # DH = DH.T
            # DHBH_T = np.dot(DH, BH_T)
        self.length = len(self.u_id)
        self.shuffle = shuffle



    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # 打乱session item_seq&price_seq的顺序
            self.u_id = self.u_id[shuffled_arg]
            self.last_id = self.last_id[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        pad_seq, num_node = [], []
        inp = self.last_id[index]

        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        # session_len = []
        # reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            # session_len.append([len(nonzero_elems)])

            pad_seq.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
                # reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.u_id[index], pad_seq, mask, self.targets[index]-1


