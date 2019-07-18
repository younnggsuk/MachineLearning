import tensorflow as tf

# Training Data set (Only Training)
x_train = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
     [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_train = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
     [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Test Data set (Only Test)
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

logits = tf.matmul(x, w) + b
H = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(H, 1)
is_correct = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, w_val, _ = sess.run([cost, w, optimizer],
                                      feed_dict={x: x_train,
                                                 y: y_train})
        if step % 20 == 0:
            print(step, cost_val)
            print(w_val)

    # Test data로 예측이 맞는지 평가
    print("\nPrediction: [{}]".format(sess.run(prediction,
                                             feed_dict={x: x_test})))
    # Test data의 결과 정확도 출력
    print("Accuracy: [{:.2%}]".format(sess.run(accuracy,
                                           feed_dict={x: x_test,
                                                      y: y_test})))


