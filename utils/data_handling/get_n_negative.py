import random
import csv
from scipy.sparse import csr_matrix
import numpy as np 
import pandas as pd 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = '../../data/dataset_1/Office_Products')
    parser.add_argument('--n_negative', type = int, default = '4')
    return parser.parse_args()

def get_n_negative(filename, n_negative):
    df = pd.read_csv(filename + '.csv', names = ['userid', 'itemid', 'ratings'])
    # 用户数和项目数
    n_users = df.userid.unique().shape[0]
    n_items = df.itemid.unique().shape[0]
    print(n_users, n_items)

    # 遍历数据，生成稀疏矩阵形式
    row, col, rating = [], [], []
    for line in df.itertuples():
        row.append(line[1])
        col.append(line[2])
        rating.append(line[3])
    matrix = csr_matrix((rating, (row, col)), shape = (n_users, n_items))
    
    all_items = set(np.arange(n_items))
    negative_dict = {}
    for i in range(n_users):
        negative_dict[i] = list(all_items - set(matrix.getrow(i).nonzero()[1]))
    test_csv = csv.reader((open(filename + '_test.csv', 'r')))
    test_userid, test_itemid, test_rating = [], [], []
    for line in test_csv:
        test_userid.append(int(line[0]))
        test_itemid.append(int(line[1]))
        test_rating.append(float(line[2]))
    print(len(test_userid))
    
    with open(filename + '_negative.csv', 'w', newline = '') as negative_out:
        negative_writer = csv.writer(negative_out, dialect = 'excel')
        for test in zip(test_userid, test_itemid, test_rating):
            row_negative = []
            row_negative.append(test)
            row_negative += random.sample(negative_dict[test[0]], n_negative)
            negative_writer.writerow(row_negative)


    
if __name__ == '__main__':
    args = parse_args()
    get_n_negative(args.filename, args.n_negative)
