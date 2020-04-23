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
        w1, w2, w3, dense, b1, b2, b3, b4 = sess.run(["w1:0", "w2:0", "w3:0", "dense:0", "b1:0", "b2:0", "b3:0", "b4:0"])
        #print(U, V)
    return w1, w2, w3, dense, b1, b2, b3, b4

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
    load_data('/home/zhanchao/RecommendationSystem/data/dataset_4/Automotive')
    Us, Vs = lv.load_variable('../model/mf_s/s.meta', '../model/mf_s')
    Ut, Vt = lv.load_variable('../model/mf_t/t.meta', '../model/mf_t')
    w1, w2, w3, dense, b1, b2, b3, b4 = load_variable('../model/mlp/mlp.meta', '../model/mlp')
    

    userid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'userid')
    itemid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'itemid')

    _Us = tf.constant(Us)
    _Vt = tf.constant(Vt)
    _w1 = tf.constant(w1)
    _w2 = tf.constant(w2)
    _w3 = tf.constant(w3)
    _dense = tf.constant(dense)
    _b1 = tf.constant(b1)
    _b2 = tf.constant(b2)
    _b3 = tf.constant(b3)
    _b4 = tf.constant(b4)

    src_user_latent_factor = tf.nn.embedding_lookup(_Us, userid)
    dst_item_latent_factor = tf.nn.embedding_lookup(_Vt, itemid)

    hidden1 = tf.nn.tanh(tf.matmul(src_user_latent_factor, w1) + b1)
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)
    hidden3 = tf.nn.tanh(tf.matmul(hidden2, w3) + b3)
    pred_dst_user_latent_factor = tf.nn.sigmoid(tf.matmul(hidden3, dense) + b4)

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

