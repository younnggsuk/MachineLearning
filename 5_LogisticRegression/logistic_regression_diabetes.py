import tensorflow as tf
import numpy as np

# 파일로부터 Dataset 받아오기
data = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.sigmoid(tf.matmul(x, w)+b)
cost = -tf.reduce_mean(y*tf.log(H) + (1-y)*tf.log(1-H))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {x: x_data, y: y_data}

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict=feed)

        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([H, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis\n", h)
    print("\nPredicted\n", p)
    print("\nAccuracy\n", a)

