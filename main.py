import argparse
import pickle
import time
from util import Data, split_validation, init_seed
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Beauty', help='dataset name: Beauty/Cell_Phones_and_Accessories/Clothing_Shoes_and_Jewelry/Yelp')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size 512')
parser.add_argument('--embSize', type=int, default=128, help='whole user/item emb size')
parser.add_argument('--chunk_embSize', type=int, default=16, help='attr, item, user, opinion')
parser.add_argument('--ui_layer', type=float, default=1, help='the number of aggregating item/user embedding')
parser.add_argument('--graph_layer', type=float, default=1, help='the number of aggregating item/user embedding with graph')
parser.add_argument('--whole_layer', type=float, default=1, help='the number of aggregating co-occurren item/user embedding')
parser.add_argument('--attr_num', type=int, default=7, help='the number of attributes')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)

def main():
    # list[0]:user_id; list[1]:x{t-1}; list[2]:label; list[3]-end: attr_coo_matrix -> user-item;
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    # list[0]:user_id; list[1]:x{t-1}; list[2]:label;
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    # init_seed(2023, True)

    if opt.dataset == 'best':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 924
        n_user = 507
        n_attr = 2
        # opin emb table 要在最前面加一行0向量
        n_opi = 1000+1
    elif opt.dataset == 'Beauty':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 10176
        n_user = 15152
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 533431
    elif opt.dataset == 'Beauty_4':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 11895
        n_user = 21584
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 4
    elif opt.dataset == 'Cell_Phones_and_Accessories':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 6208
        n_user = 7598
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 275529
    elif opt.dataset == 'Clothing_Shoes_and_Jewelry':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 12538
        n_user = 9847
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 408222
    elif opt.dataset == 'Home_and_Kitchen':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 16597
        n_user = 18300
        n_attr = 9
        # opin emb table 要在最前面加一行0向量
        n_opi = 736772
    elif opt.dataset == 'Home_and_Kitchen':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 16597
        n_user = 18300
        n_attr = 9
        # opin emb table 要在最前面加一行0向量
        n_opi = 736772
    elif opt.dataset == 'Sports_and_Outdoors':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 11017
        n_user = 11817
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 386944
    elif opt.dataset == 'Yelp':
        # 不需要+1，后来拼接一个0 embedding；需要+1，后来用1序的数组获得对应表示，不然数据越界
        n_item = 12391
        n_user = 12373
        n_attr = 7
        # opin emb table 要在最前面加一行0向量
        n_opi = 567599
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item matrix_pv
    train_data = Data(train_data, shuffle=True, is_train=True, n_attr=n_attr, n_user=n_user, n_item=n_item)
    test_data = Data(test_data, shuffle=True)
    model = trans_to_cuda(FineRec(user_adj = train_data.user_adj, item_adj = train_data.item_adj, ui_list = train_data.ui_list, iu_list = train_data.iu_list, user_io= train_data.user_io, item_uo = train_data.item_uo, n_attr=n_attr, n_opi=n_opi, n_user=n_user, n_item=n_item, lr=opt.lr, ui_layer=opt.ui_layer, graph_layer =opt.graph_layer,  whole_layer=opt.whole_layer, attr_num=opt.attr_num, l2=opt.l2, dataset=opt.dataset, num_heads=opt.num_heads, emb_size=opt.embSize, chunk_embSize=opt.chunk_embSize, batch_size=opt.batchSize))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        # for K in top_K:
        #     print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f\tEpoch: %d,  %d, %d' %
        #           (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],K, best_results['metric%d' % K][2],
        #            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
