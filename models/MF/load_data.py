import pandas as pd 
import numpy as np 

from scipy.sparse import csr_matrix

def load_data(path = "../../data/dataset_1/Office_Products", header = ['userid', 'itemid', 'rating']):
    df = pd.read_csv(path + '.csv', names = header)
    n_users = df.userid.unique().shape[0]
    n_items = df.itemid.unique().shape[0]
    #print(n_users, n_items)

    train_df = pd.read_csv(path + '_train.csv', names = header)
    train_n_users = train_df.userid.unique().shape[0]
    train_n_items = train_df.itemid.unique().shape[0]
    #print(train_n_users, train_n_items)

    test_df = pd.read_csv(path + '_test.csv', names = header)
    test_n_users = test_df.userid.unique().shape[0]
    test_n_items = test_df.itemid.unique().shape[0]
    #print(test_n_users, test_n_items)

    # 载入train数据
    train_row, train_col, train_rating = [], [], []

    for line in train_df.itertuples():
        train_row.append(line[1])
        train_col.append(line[2])
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_row)), shape = (n_users, n_items))

    # 载入test数据
    test_row, test_col, test_rating = [], [], []

    for line in test_df.itertuples():
        test_row.append(line[1])
        test_col.append(line[2])
        test_rating.append(line[3])
    
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape = (n_users, n_items))

    print("Load data finished. Number of users: %d, Number of items: %d" %(n_users, n_items))

    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

if __name__ == '__main__':
    load_data()