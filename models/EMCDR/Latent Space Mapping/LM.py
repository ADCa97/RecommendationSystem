import numpy as np 
import tensorflow as tf 
import os
import load_variable as lv 
from tensorflow.contrib import layers

def linear_mapping(input_Us, input_Ut, beta, learning_rate, training_epochs, display_step=100):
    '''线性映射函数
    input: 
        input_Vs(ndarray): 源领域矩阵
        input_Vt(ndarray): 目标领域矩阵
        beta(float): 正则化参数
        learning_rate(float): 学习率
        training_epochs(int): 最大迭代次数
        display_step(int): 展示步数
    output: 
        M, b: 映射函数参数
    '''
    k, m = np.shape(input_Us)

    # 1. 设置变量
    Us = tf.placeholder(tf.float32, [k, m])
    Ut = tf.placeholder(tf.float32, [k, m])
    M = tf.Variable(tf.random_normal([m, m], stddev = 0.01), name = 'M')
    b = tf.Variable(tf.zeros([m]), name="b")

    # 2. 构造模型
    predUt = tf.matmul(Us, M) + b
    regM = layers.l2_regularizer(beta)(M)
    cost = tf.reduce_mean(tf.square(Ut - predUt)) + regM
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # 3. 开始训练
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(train_step, feed_dict={Us: input_Us, Ut: input_Ut})

            if (epoch + 1) % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={Us: input_Us, Ut: input_Ut})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        
        # 打印变量
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
            print(v)
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, "../model/lm/lm")
        print("Optimization Finished!")


if __name__ == '__main__':
    Us, Vs = lv.load_variable('../model/mf_s/s.meta', '../model/mf_s')
    Ut, Vt = lv.load_variable('../model/mf_t/t.meta', '../model/mf_t')
    beta = 0.0001
    learning_rate = 0.1
    training_epochs = 100000
    display_step = 10
    linear_mapping(Us, Ut, beta, learning_rate, training_epochs, display_step)
    print(tf.global_variables())