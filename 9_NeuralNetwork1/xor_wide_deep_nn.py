import tensorflow as tf
import numpy as np

# XOR data set
xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yData = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# Wide(10) and Deep(4 layer) Neural Network
w1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
h = tf.sigmoid(tf.matmul(layer3, w4) + b4)

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