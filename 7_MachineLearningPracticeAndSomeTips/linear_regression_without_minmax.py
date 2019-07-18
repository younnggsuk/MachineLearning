import tensorflow as tf
import numpy as np

data = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = data[:, 0:-1]
y_data = data[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(H - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={x: x_data,y: y_data})

        # Cost와 Hypothesis(예측 값)이 밖으로 튕겨나가서 Nan이 됨
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}),
                        sess.run(H, feed_dict={x: x_data, y: y_data}))





