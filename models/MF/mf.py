import argparse


import tensorflow as tf 
import numpy as np 
import pandas as pd 
import time
import load_data as ld

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type = str, default = '../../data/dataset_1/Office_Products')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--reg_rate', nargs = '?', default = '[0.001, 1]')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 128)
    
    return parser.parse_args()

class MF():
    def __init__(self, num_user, num_item, learning_rate = 0.001, reg_rate = 0.01, epochs = 50, batch_size = 128):
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epochs
        self.batch_size = batch_size
        print("MF init.", self.reg_rate)
    
    def build_network(self, num_factor = 30):
        self.user_id = tf.placeholder(dtype = tf.int32, shape = [None], name = 'user_id')
        self.item_id = tf.placeholder(dtype = tf.int32, shape = [None], name = 'item_id')
        self.y = tf.placeholder(dtype = tf.float32, shape = [None], name = 'rating')
        
        self.U = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev = 0.01), name = 'User')
        self.V = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev = 0.01), name = 'Item')

        user_latent_factor = tf.nn.embedding_lookup(self.U, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.V, self.item_id)

        self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1)
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) + self.reg_rate[0] * tf.nn.l2_loss(self.U) + self.reg_rate[1] * tf.nn.l2_loss(self.V)
if __name__ == '__main__':
    args = parse_args()
    datapath = args.datapath
    learning_rate = args.learning_rate
    reg_rate = eval(args.reg_rate)
    epochs = args.epochs
    batch_size = args.batch_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_data, test_data, n_user, n_item = ld.load_data(path = datapath)
    with tf.Session(config = config) as sess:
        
        t = train_data.tocoo()
        user = t.row.reshape(-1)
        item = t.col.reshape(-1)
        rating = t.data
        # 构建图
        model = None
        model = MF(n_user, n_item, learning_rate = learning_rate, reg_rate = reg_rate, epochs = epochs, batch_size = batch_size)        
        if model is not None:
            model.build_network()

        optimizer = tf.train.AdamOptimizer(learning_rate = model.learning_rate).minimize(model.loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        #开始训练
        for epoch in range(model.epochs):
            print('Epoch: %04d' %(epoch))
            # 每一轮训练分为若干batch
            num_training = len(rating)
            total_batch = int(num_training / model.batch_size)

            idxs = np.random.permutation(num_training)

            user_random = list(user[idxs])
            item_random = list(item[idxs])
            rating_random = list(rating[idxs])

            for i in range(total_batch):
                batch_user = user_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_item = item_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_rating = rating_random[i * model.batch_size : (i + 1) * model.batch_size]
                _, loss = sess.run([optimizer, model.loss], feed_dict = {model.user_id: batch_user,
                                                                         model.item_id: batch_item,
                                                                         model.y: batch_rating})
                if i % 1000 == 0:
                    print("cost = %.9f" %(loss))
            if epoch % 10 == 0:
                error = 0
                test_set = list(test_data.keys())
                for (u, i) in test_set:
                    pred_rating_test = sess.run(model.pred_rating, feed_dict = {model.user_id: [u],
                                                                                  model.item_id: [i]})
                    
                    error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
                print("error = %.9f number = %04d RMSE = %.9f" %(error, len(test_set), np.sqrt(error / len(test_set))))
