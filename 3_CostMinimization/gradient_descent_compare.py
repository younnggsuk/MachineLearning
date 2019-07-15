import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# Weight를 5.0에서부터 시작
W = tf.Variable(5.0)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# optimizer를 통한 gradient
compute_gradient = optimizer.compute_gradients(cost)

# 직접 계산한 gradient
manual_gradient = 0.1 * tf.reduce_mean((W * X - Y) * X)

# optimizer를 통해 얻은 gradient를 적용
apply_compute = optimizer.apply_gradients(compute_gradient)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 20):
    # step, W값, [직접 계산한 gradient, optimizer로 계산한 gradient] 형태로 출력
    print(step, sess.run(W), sess.run([manual_gradient, compute_gradient]))
    sess.run(apply_compute)