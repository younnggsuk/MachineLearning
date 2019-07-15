import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient 식
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient

# 위 식을 통해 나온 Weight로 update
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# W 값 찾아가는 과정 출력
for step in range(1, 21):
    sess.run(update)
    print(step, sess.run(cost), sess.run(W))