import argparse
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import time 
import load_data as ld 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, default = '../../data/dataset_1/Office_Products')
    parser.add_argument('--dst', type = str, default = '../../data/dataset_1/Movies_and_TV')
    parser.add_argument('--share', type = str, default = '../../data/dataset_1/share_info.csv')
    parser.add_argument('--savepath', type = str, default = 'default.csv')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--reg_rate', nargs = '?', default = '[0.001, 0.001, 0.001, 0.001, 0.001, 0.001]')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 128)
    return parser.parse_args()
    
class FinalModel():
    def __init__(self, num_user, src_num_item, dst_num_item, learning_rate = 0.001, reg_rate = [0, 0, 0], epochs = 50, batch_size = 128):
        self.num_user = num_user
        self.src_num_item = src_num_item
        self.dst_num_item = dst_num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epochs
        self.batch_size = batch_size
        print("UMF init")
        pass 
    def build_network(self, num_factor = 50):
        self.shareuser = tf.placeholder(dtype = tf.int32, shape = [None], name = 'shareuser')
        self.src_itemid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'src_itemid')
        self.src_ratings = tf.placeholder(dtype = tf.float32, shape = [None], name = 'src_ratings')
        self.dst_itemid = tf.placeholder(dtype = tf.int32, shape = [None], name = 'dst_itemid')
        self.dst_ratings = tf.placeholder(dtype = tf.float32, shape = [None], name = 'dst_ratings')

        #self.U = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev = 0.01), name = 'ShareUser')
        self.src_U = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev = 0.01), name = 'Src_User')
        self.dst_U = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev = 0.01), name = 'Dst_User')
        self.src_V = tf.Variable(tf.random_normal([self.src_num_item, num_factor], stddev = 0.01), name = 'Src_Item')
        self.dst_V = tf.Variable(tf.random_normal([self.dst_num_item, num_factor], stddev = 0.01), name = 'Dst_Item')

        src_user_latent_factor = tf.nn.embedding_lookup(self.src_U, self.shareuser)
        dst_user_latent_factor = tf.nn.embedding_lookup(self.dst_U, self.shareuser)
        src_item_latent_factor = tf.nn.embedding_lookup(self.src_V, self.src_itemid)
        dst_item_latent_factor = tf.nn.embedding_lookup(self.dst_V, self.dst_itemid)


        # 方案一
        mlp_vector = tf.concat([src_user_latent_factor, dst_user_latent_factor], 1)
        self.W1 = tf.Variable(tf.random_normal([2 * num_factor, num_factor], stddev = 0.01), name = 'W1')
        self.B1 = tf.Variable(tf.zeros([num_factor]))
        self.W2 = tf.Variable(tf.random_normal([num_factor, num_factor], stddev = 0.01), name = 'W2')
        self.B2 = tf.Variable(tf.zeros([num_factor]))
        self.W3 = tf.Variable(tf.random_normal([num_factor, num_factor], stddev = 0.01), name = 'W3')
        self.B3 = tf.Variable(tf.zeros([num_factor]))
        self.Dense = tf.Variable(tf.random_normal([num_factor, num_factor], stddev = 0.01), name = 'Dense')
        self.B4 = tf.Variable(tf.zeros([num_factor]))

        #self.Dense = tf.Variable(tf.random_normal([int(num_factor / 4), 1], stddev = 0.01), name = 'Dense')

        hidden1 = tf.nn.relu(tf.matmul(mlp_vector, self.W1) + self.B1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.W2) + self.B2)
        hidden3 = tf.nn.relu(tf.matmul(hidden2, self.W3) + self.B3)
        output = tf.nn.softmax(tf.matmul(hidden3, self.Dense) + self.B4)
        #output = tf.nn.softmax(tf.matmul(hidden3, self.Dense))
        #print(output)

        

        pred_dst_user_latent_factor = src_user_latent_factor + output
        self.src_pred_rating = tf.reduce_sum(tf.multiply(src_user_latent_factor, src_item_latent_factor), 1)
        self.dst_pred_rating = tf.reduce_sum(tf.multiply(pred_dst_user_latent_factor, dst_item_latent_factor), 1)

        self.loss_Src_Dst_U = tf.nn.l2_loss(pred_dst_user_latent_factor - dst_user_latent_factor)
        self.loss = tf.reduce_sum(tf.square(self.src_ratings - self.src_pred_rating)) + \
                    tf.reduce_sum(tf.square(self.dst_ratings - self.dst_pred_rating)) + \
                    self.reg_rate[0] * tf.nn.l2_loss(self.src_U) + \
                    self.reg_rate[1] * tf.nn.l2_loss(self.dst_U) + \
                    self.reg_rate[2] * tf.nn.l2_loss(self.src_V) + \
                    self.reg_rate[3] * tf.nn.l2_loss(self.dst_V) + \
                    self.reg_rate[4] * self.loss_Src_Dst_U + \
                    self.reg_rate[5] * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3)) + tf.nn.l2_loss(self.Dense)
                    #self.reg_rate[0] * tf.nn.l2_loss(hidden2)
        print("FinalModel build_network")



if __name__ == '__main__':
    args = parse_args()
    src = args.src
    dst = args.dst
    share = args.share
    learning_rate = args.learning_rate
    reg_rate = eval(args.reg_rate)
    epochs = args.epochs
    batch_size = args.batch_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    src_test_matrix, src_n_users, src_n_items = ld.get_test_csr_matrix(src)
    dst_test_matrix, dst_n_users, dst_n_items = ld.get_test_csr_matrix(dst)
    shareuser, src_itemid, src_ratings, dst_itemid, dst_ratings = ld.get_train_list(share)
    print(src_n_users, src_n_items, dst_n_users, dst_n_items, len(shareuser), type(shareuser))
    with tf.Session(config = config) as sess:
        model = None
        model = FinalModel(src_n_users, src_n_items, dst_n_items, learning_rate, reg_rate, epochs, batch_size)
        if model is not None:
            model.build_network()
        optimizer = tf.train.AdamOptimizer(learning_rate = model.learning_rate).minimize(model.loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        cost = []
        src_rmse = []
        dst_rmse = []
        all_rmse = []
        for epoch in range(model.epochs):
            print("Epoch: %04d" %epoch)

            # 每一轮训练分为若干batch
            num_training = len(shareuser)
            total_batch = int(num_training / model.batch_size)

            idxs = np.random.permutation(num_training)

            shareuser_random = list(shareuser[idxs])
            src_itemid_random = list(src_itemid[idxs])
            src_ratings_random = list(src_ratings[idxs])
            dst_itemid_random = list(dst_itemid[idxs])
            dst_ratings_random = list(dst_ratings[idxs])
            average_cost = []
            for i in range(total_batch):
                batch_shareuser = shareuser_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_src_itemid = src_itemid_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_src_ratings = src_ratings_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_dst_itemid = dst_itemid_random[i * model.batch_size : (i + 1) * model.batch_size]
                batch_dst_ratings = dst_ratings_random[i * model.batch_size : (i + 1) * model.batch_size]

                _, loss = sess.run([optimizer, model.loss], feed_dict = {model.shareuser: batch_shareuser,
                                                                         model.src_itemid: batch_src_itemid, 
                                                                         model.src_ratings: batch_src_ratings,
                                                                         model.dst_itemid: batch_dst_itemid,
                                                                         model.dst_ratings: batch_dst_ratings})
                average_cost.append(loss)
                '''
                if i % 1000 == 0:
                    print("Cost = %.9f" %loss)
                '''
            if epoch % 5 == 0:
                all_error = 0
                all_test = 0
                src_error = 0
                src_set = list(src_test_matrix.keys())
                for (u, i) in src_set:
                    src_pred_rating_test = sess.run(model.src_pred_rating, feed_dict = {model.shareuser: [u],
                                                                                    model.src_itemid: [i]})
                    src_error += (float(src_test_matrix.get((u, i))) - src_pred_rating_test[0]) ** 2
                print("src_error = %.9f number = %04d RMSE = %.9f" %(src_error, len(src_set), np.sqrt(src_error / len(src_set))))
                all_error += src_error
                all_test += len(src_set)

                dst_error = 0
                dst_set = list(dst_test_matrix.keys())
                for (u, i) in dst_set:
                    dst_pred_rating_test = sess.run(model.dst_pred_rating, feed_dict = {model.shareuser: [u],
                                                                                    model.dst_itemid: [i]})
                    dst_error += (float(dst_test_matrix.get((u, i))) - dst_pred_rating_test[0]) ** 2
                print("dst_error = %.9f number = %04d RMSE = %.9f" %(dst_error, len(dst_set), np.sqrt(dst_error / len(dst_set))))
                all_error += dst_error
                all_test += len(dst_set)
                print("all_error = %.9f number = %04d RMSE = %.9f" %(all_error, all_test, np.sqrt(all_error / all_test)))
                cost.append(np.mean(average_cost))
                src_rmse.append(np.sqrt(src_error / len(src_set)))
                dst_rmse.append(np.sqrt(dst_error / len(dst_set)))
                all_rmse.append(np.sqrt(all_error / all_test))
        dict = {'cost':cost, 'src_rmse':src_rmse, 'dst_rmse':dst_rmse, 'all_rmse':all_rmse}
        df = pd.DataFrame(dict)
        df.to_csv(args.savepath, header = 0)