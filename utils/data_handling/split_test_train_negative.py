import pandas as pd
import csv
from scipy.sparse import csr_matrix
import numpy as np 
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = '../../data/dataset_1/Office_Products')
    return parser.parse_args()

def split_test_train_negative(filename):
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

    n_negative = 99

    all_test_userid, all_test_itemid, all_test_rating = [], [], []
    all_train_userid, all_train_itemid, all_train_rating = [], [], []


    with open(filename + '_negative.csv', 'w', newline = '') as negative_out:
        negative_writer = csv.writer(negative_out, dialect = 'excel')
        for i in range(n_users):
            row_i = matrix.getrow(i).tocoo()
            # userid、itemid、rating是用户i和项目的交互数据
            userid = np.array([i] * len(row_i.row.reshape(-1)))
            itemid = row_i.col.reshape(-1)
            rating = row_i.data
            # 用户i和项目的交互数量
            n = len(userid)
            #print(n)
            # 10%的比例抽取测试集
            n_test = math.ceil(n * 0.1)
            # 按idxs随机抽取对应的用户评分数据
            idxs = np.random.permutation(n)
            idxs_test = idxs[:n_test]
            idxs_train = idxs[n_test:]

            test_userid, test_itemid, test_rating = userid[idxs_test], itemid[idxs_test], rating[idxs_test]

            train_userid, train_itemid, train_rating = userid[idxs_train], itemid[idxs_train], rating[idxs_train]

            all_test_userid += list(test_userid)
            all_test_itemid += list(test_itemid)
            all_test_rating += list(test_rating)

            all_train_userid += list(train_userid)
            all_train_itemid += list(train_itemid)
            all_train_rating += list(train_rating)

            # 为每一个(test_userid, test_itemid, test_rating)生成n_negative个数据
            #print(list(zip(test_userid, test_itemid, test_rating)))
            for test in zip(test_userid, test_itemid, test_rating):
                row_negative = []
                row_negative.append(test)
                # 接下来随机生成n_negative个negative数据
                j = 0
                while j < n_negative:
                    k = np.random.randint(n_items)
                    if k not in itemid:
                        row_negative.append(k)
                        j += 1
                negative_writer.writerow(row_negative)
    dict = {'userid': all_test_userid, 'itemid': all_test_itemid, 'rating': all_test_rating}
    pd.DataFrame.from_dict(dict).to_csv(filename + '_test.csv', index = False, header = False)

    dict = {'userid': all_train_userid, 'itemid': all_train_itemid, 'rating': all_train_rating}
    pd.DataFrame.from_dict(dict).to_csv(filename + '_train.csv', index = False, header = False)

if __name__ == '__main__':
    args = parse_args()

    split_test_train_negative(args.filename)
#print(list(zip(all_test_userid, all_test_itemid, all_test_rating)))











#print(matrix.getrow(0).todok().keys())
#print(matrix.getrow(0).tocoo().row.reshape(-1))
