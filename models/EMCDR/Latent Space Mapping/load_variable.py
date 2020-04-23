import numpy as np 
import tensorflow as tf 


def load_variable(meta_path, ckpt_path):
    new_graph = tf.Graph()
    with tf.Session(graph = new_graph) as sess:
        loader = tf.train.import_meta_graph(meta_path)
        loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        U, V = sess.run(["User:0", "Item:0"])
        #print(U, V)
    return U, V 
if __name__ == '__main__':
    load_variable('/home/zhanchao/RecommendationSystem/models/EMCDR/model/mf_s/s.meta', '/home/zhanchao/RecommendationSystem/models/EMCDR/model/mf_s')