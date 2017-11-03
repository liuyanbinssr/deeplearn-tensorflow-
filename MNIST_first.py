import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#导入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#初始化参数
W = tf.Variable(tf.zeros([mnist.train.images.shape[1],10])) #W初始化为0
b = tf.Variable(tf.zeros([10])) #b初始化为0
#建立模型
x = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]]) 

y = tf.nn.softmax(tf.matmul(x,W) + b) #softmax激活

y_ = tf.placeholder("float", [None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()#初始化变量
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))