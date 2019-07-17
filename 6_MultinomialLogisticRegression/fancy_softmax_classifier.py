import tensorflow as tf
import numpy as np

# File에서 Data set 가져오기
data = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

num_class = 7

x = tf.placeholder(tf.float32, shape=[None, 16])
y = tf.placeholder(tf.int32, shape=[None, 1])

# y data를 one_hot 형태로 변환
# [[[], []]] 이런 형태로 반환됨
y_one_hot = tf.one_hot(y, num_class)
# [[], []] 형태로 바꾸기 위해 reshape 사용
y_one_hot = tf.reshape(y_one_hot, [-1, num_class])

w = tf.Variable(tf.random_normal([16, num_class]), name='weight')
b = tf.Variable(tf.random_normal([num_class]), name='bias')

logits = tf.matmul(x, w) + b
H = tf.nn.softmax(logits)

# Cost 함수에 Cross Entropy 적용
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측 값과 실제 값의 비교 연산들
prediction = tf.argmax(H, 1)
isTruePredict = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(isTruePredict, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={x: x_data,
                                       y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: x_data,
                                                              y: y_data})
            print("Step:{:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step, loss, acc))

    # 예측 값과 실제 값의 비교 출력
    pred = sess.run(prediction, feed_dict={x: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}]\tPrediction:{}\tTrue Y: {}".format((p == y), int(p), int(y)))