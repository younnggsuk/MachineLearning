import tensorflow as tf

# Data set
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

num_class = 3

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, num_class])

w = tf.Variable(tf.random_normal([4, num_class]), name='weight')
b = tf.Variable(tf.random_normal([num_class]), name='bias')

# Hypothesis using Softmax
H = tf.nn.softmax(tf.matmul(x, w) + b)

# Cross-Entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(H), axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train, feed_dict={x: x_data,
                               y: y_data})

    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={x: x_data,
                                              y: y_data}))

# Hypothesis로 구한 값 중 가장 큰 값의 위치 반환
res = sess.run(tf.arg_max(sess.run(H, feed_dict={x: [[1, 11, 7, 9],
                                                     [1, 3, 4, 3],
                                                     [1, 1, 0, 1]]}), 1))
print(res)