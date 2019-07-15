import tensorflow as tf

# data set X, Y를 placeholder로 생성
# shape : 1차원, element 수 지정 X (None)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# placeholder에 feed_dict로 training dataset 적용
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:[1, 2, 3, 4, 5],
                            Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 잘 학습 됐는지 확인
print("result :", sess.run(hypothesis, feed_dict={X:[10, 20, 30]}))