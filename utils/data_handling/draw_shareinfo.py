import pandas as pd 
import csv
from scipy.sparse import csr_matrix
import numpy as np 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, default = '../../data/dataset_1/Office_Products')
    parser.add_argument('--dst', type = str, default = '../../data/dataset_1/Movies_and_TV')
    parser.add_argument('--save', type = str, default = '../../data/dataset_1/share_info.csv')
    return parser.parse_args()

def get_train_csr_matrix(filename):
    df = pd.read_csv(filename + '.csv', names = ['userid', 'itemid', 'rating'])
    n_users = df.userid.unique().shape[0]
    n_items = df.itemid.unique().shape[0]
    #print(n_users, n_items)

    train_df = pd.read_csv(filename + '_train.csv', names = ['userid', 'itemid', 'rating'])
    train_n_users = train_df.userid.unique().shape[0]
    train_n_items = train_df.itemid.unique().shape[0]
    #print(train_n_users, train_n_items)

    # 载入train数据
    train_row, train_col, train_rating = [], [], []

    for line in train_df.itertuples():
        train_row.append(line[1])
        train_col.append(line[2])
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape = (n_users, n_items))

    #print(train_matrix.shape[0])
    return train_matrix

# <shareuser, src_item,id src_rating, dst_itemid, dst_rating>
if __name__ == '__main__':
    args = parse_args()
    src_matrix = get_train_csr_matrix(args.src)
    dst_matrix = get_train_csr_matrix(args.dst)
    #print(src_matrix.nnz)
    user = src_matrix.tocoo().row.reshape(-1)
    #print(len(user))

    n_users = src_matrix.shape[0]


    shareuser, src_itemid, src_ratings, dst_itemid, dst_ratings = [], [], [], [], []
    for u in range(n_users):
        src_row = src_matrix.getrow(u).tocoo()
        src_items = src_row.col.reshape(-1)
        src_rating = src_row.data
        
        dst_row = dst_matrix.getrow(u).tocoo()
        dst_items = dst_row.col.reshape(-1)
        dst_rating = dst_row.data
        print(len(src_items), len(dst_items))
        for i in range(len(src_items)):
            for j in range(len(dst_items)):
                shareuser += [u]
                src_itemid.append(src_items[i])
                src_ratings.append(src_rating[i])

                dst_itemid.append(dst_items[j])
                dst_ratings.append(dst_rating[j])
    dict = {'shareuser': shareuser, 'src_itemid': src_itemid, 'src_ratings': src_ratings, 'dst_itemid': dst_itemid, 'dst_ratings': dst_ratings}
    pd.DataFrame.from_dict(dict).to_csv(args.save, index = False, header = False)


    