import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# Weight를 5.0에서부터 시작
W = tf.Variable(5.0)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 20):
    sess.run(train)
    print(step, sess.run(cost), sess.run(W))