import tensorflow as tf

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(H - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 2001):
    cost_val, H_val, _ = sess.run([cost, H, train],
                                  feed_dict={x: x_data,
                                             y: y_data})
    if step % 20 == 0:
        print(step, cost_val, H_val)