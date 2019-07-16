import tensorflow as tf
import numpy as np

# Get data set from file
data = np.loadtxt('test-score1.csv', dtype=np.float32, delimiter=',')
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(H-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 2001):
    cost_val, H_val, _ = sess.run([cost, H, train],
                                  feed_dict={x: x_data,
                                             y: y_data})
    if step % 20 == 1 :
        print(cost_val)
        print(H_val)

# 위의 학습 모델로 점수 예측
print("Score 1:", sess.run(H, feed_dict={x: [[100, 70, 101]]}))
print("Score 2:", sess.run(H, feed_dict={x: [[60, 70, 110]]}))