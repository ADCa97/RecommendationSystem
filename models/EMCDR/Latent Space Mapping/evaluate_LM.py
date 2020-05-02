import tensorflow as tf 
import numpy as np 
import load_variable as lv 
import pandas as pd 
from scipy.sparse import csr_matrix

def load_variable(meta_path, ckpt_path):
    new_graph = tf.Graph()
    with tf.Session(graph = new_graph) as sess:
        loader = tf.train.import_meta_graph(meta_path)
        loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        M, b = sess.run(["M:0", "b:0"])
        #print(U, V)
    return M, b 

def load_data(path = "../../data/dataset_1/Office_Products", header = ['userid', 'itemid', 'rating']):
    df = pd.read_csv(path + '.csv', names = header)
    n_users = df.userid.unique().shape[0]
    n_items = df.itemid.unique().shape[0]
    print(n_users, n_items)


    test_df = pd.read_csv(path + '_test.csv', names = header)
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

    print("Load data finished. Number of users: %d, Number of items: %d" %(n_users, n_items))

    return test_matrix.todok(), n_users, n_items

if __name__ == '__main__':
    #load_data('/home/zhanchao/RecommendationSystem/data/dataset_4/Automotive')
    Us, Vs = lv.load_variable('../model/mf_s/s.meta', '../model/mf_s')
    Ut, Vt = lv.load_variable('../model/mf_t/t.meta', '../model/mf_t')
    M, b = load_variable('../model/lm/lm.meta', '../model/lm')
    

    userid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'userid')
    itemid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'itemid')

    _Us = tf.constant(Us)
    _Vt = tf.constant(Vt)
    _M = tf.constant(M)
    _b = tf.constant(b)

    src_user_latent_factor = tf.nn.embedding_lookup(_Us, userid)
    dst_item_latent_factor = tf.nn.embedding_lookup(_Vt, itemid)
    pred_dst_user_latent_factor = tf.matmul(src_user_latent_factor, _M) + _b 
    pred = tf.reduce_sum(tf.multiply(pred_dst_user_latent_factor, dst_item_latent_factor), 1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    test_data, n_user, n_item = load_data('/home/zhanchao/RecommendationSystem/data/dataset_4/Automotive')

    with tf.Session(config = config) as sess:
        error = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = sess.run(pred, feed_dict = {userid: [u], itemid: [i]})
            
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
        print("error = %.9f number = %04d RMSE = %.9f" %(error, len(test_set), np.sqrt(error / len(test_set))))

