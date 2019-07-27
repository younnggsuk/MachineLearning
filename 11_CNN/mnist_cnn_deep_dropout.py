import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])
xImg = tf.reshape(x, [-1, 28, 28, 1])

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
l1 = tf.nn.conv2d(xImg, w1, strides=[1, 1, 1, 1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
# output : [14 * 14], channels : 32

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)
# output : [7 * 7], channels : 64

w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
l3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
l3 = tf.nn.relu(l3)
l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
# output : [4 * 4], channels : 128

l3Flat = tf.reshape(l3, [-1, 4*4*128])

w4 = tf.get_variable('w4', shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
l4 = tf.matmul(l3Flat, w4) + b4
l4 = tf.nn.relu(l4)
l4 = tf.nn.dropout(l4, keep_prob=keep_prob)

w5 = tf.get_variable('w5', shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))

logits = tf.matmul(l4, w5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 15
batchSize = 100

print("Learning Start!!")

for epoch in range(epochs):
    avgCost = 0
    batches = int(mnist.train.num_examples / batchSize)

    for batch in range(batches):
        batchX, batchY = mnist.train.next_batch(batchSize)
        costVal, _ = sess.run([cost, optimizer], feed_dict={x: batchX,
                                                            y: batchY,
                                                            keep_prob: 0.7})
        avgCost += (costVal/batches)

    print("Epoch: %04d" % (epoch+1), "Cost: %0.8f" % avgCost)

print("Learning Finished!!")

isCorrect = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

print("Accuracy: %0.8f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                        y: mnist.test.labels,
                                                        keep_prob: 1}))
