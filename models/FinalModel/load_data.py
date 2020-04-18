import pandas as pd 
import numpy as np 

from scipy.sparse import csr_matrix

def get_test_csr_matrix(path = '../../data/dataset_1/Office_Products'):
    df = pd.read_csv(path + '.csv', names = ['userid', 'itemid', 'rating'])
    n_users = df.userid.unique().shape[0]
    n_items = df.itemid.unique().shape[0]
    #print(n_users, n_items)

    test_df = pd.read_csv(path + '_test.csv', names = ['userid', 'itemid', 'rating'])
    test_n_users = test_df.userid.unique().shape[0]
    test_n_items = test_df.itemid.unique().shape[0]
    #print(test_n_users, test_n_items)

    # 载入test数据
    test_row, test_col, test_rating = [], [], []

    for line in test_df.itertuples():
        test_row.append(line[1])
        test_col.append(line[2])
        test_rating.append(line[3])
        
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape = (n_users, n_items))
    return test_matrix.todok(), n_users, n_items

def get_train_list(path = '../../data/dataset_1/share_info.csv'):
    df = pd.read_csv(path, names = ['shareuser', 'src_itemid', 'src_ratings', 'dst_itemid', 'dst_ratings'])
    shareuser, src_itemid, src_ratings, dst_itemid, dst_ratings = [], [], [], [], []
    for line in df.itertuples():
        shareuser.append(line[1])
        src_itemid.append(line[2])
        src_ratings.append(line[3])
        dst_itemid.append(line[4])
        dst_ratings.append(line[5])
    return np.array(shareuser), np.array(src_itemid), np.array(src_ratings), np.array(dst_itemid), np.array(dst_ratings)
if __name__ == '__main__':
    src, src_n_users, src_n_items = get_test_csr_matrix('../../data/dataset_1/Office_Products')
    print(src_n_users, src_n_items)

    dst, dst_n_users, dst_n_items = get_test_csr_matrix('../../data/dataset_1/Movies_and_TV')
    print(dst_n_users, dst_n_items)
    shareuser, src_itemid, src_ratings, dst_itemid, dst_ratings = get_train_list()
    print(shareuser)