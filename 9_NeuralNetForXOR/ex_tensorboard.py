import tensorflow as tf
import numpy as np

xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yData = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope("Layer1"):
    w1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

    tf.summary.histogram("w1", w1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    w2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
    b2 = tf.Variable(tf.random_normal([10]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

    tf.summary.histogram("w2", w2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Layer2", layer2)


with tf.name_scope("Layer3"):
    w3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

    tf.summary.histogram("w3", w3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("Layer3", layer3)


with tf.name_scope("Layer4"):
    w4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    h = tf.sigmoid(tf.matmul(layer3, w4) + b4)

    tf.summary.histogram("w4", w4)
    tf.summary.histogram("b4", b4)
    tf.summary.histogram("Hypothesis", h)


with tf.name_scope("Cost"):
    cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

    tf.summary.scalar("Cost", cost)


training = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, yData), dtype=tf.float32))

tf.summary.scalar("Accuracy", accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/log1")
writer.add_graph(sess.graph)

for step in range(10001):
    _, summaryVal = sess.run([training, merged_summary], feed_dict={x: xData, y: yData})
    writer.add_summary(summaryVal, global_step=step)

    if step % 1000 == 0:
        print(step, sess.run(cost, feed_dict={x: xData, y: yData}))


hVal, predictionVal, accuracyVal = sess.run([h, prediction, accuracy], feed_dict={x: xData, y: yData})
print("\n\nHypothesis\n{}\nPrediction\n{}\nAccuracy\n{}\n".format(hVal, predictionVal, accuracyVal))