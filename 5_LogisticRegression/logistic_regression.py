import tensorflow as tf

# Data set
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# x, y
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# Weight, bias
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid
H = tf.sigmoid(tf.matmul(x, w) + b)

# Cost
cost = -tf.reduce_mean(y*tf.log(H) + (1-y)*tf.log(1-H))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# predicted : Hypothesis가 0.5보다 크면 1, 아니면 0
predicted = tf.cast(H > 0.5, dtype=tf.float32)
# accuracy : y값과 비교한 predicted의 정확도
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data,
                                                     y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Hypothesis, 예측 결과, 정확도 출력
    h, p, a = sess.run([H, predicted, accuracy], feed_dict={x: x_data,
                                                            y: y_data})
    print("\nHypothesis\n", h)
    print("\nPredicted\n", p)
    print("\nAccuracy\n", a)
