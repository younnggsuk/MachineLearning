import tensorflow as tf
import numpy as np

# XOR data set
xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yData = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# Neural Network
# layer1 = x*w1 + b1
w1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# h = layer1*w2 + b2
w2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
h = tf.sigmoid(tf.matmul(layer1, w2) + b2)

cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))
training = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, yData), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(training, feed_dict={x: xData, y: yData})

    if step % 1000 == 0:
        print(step, sess.run(cost, feed_dict={x: xData, y: yData}))


hVal, predictionVal, accuracyVal = sess.run([h, prediction, accuracy], feed_dict={x: xData, y: yData})
print("\n\nHypothesis\n{}\nPrediction\n{}\nAccuracy\n{}\n".format(hVal, predictionVal, accuracyVal))