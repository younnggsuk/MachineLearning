import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

# one_hot 형태로 손글씨 데이터를 가져옴
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 0~9까지 이므로 class는 10개
num_class = 10

# 28*28크기의 데이터이므로 784
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, num_class])

w = tf.Variable(tf.random_normal([784, num_class]))
b = tf.Variable(tf.random_normal([num_class]))

logits = tf.matmul(x, w) + b
H = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(H, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 총 15번의 epoch
num_epochs = 15
# epoch의 크기
epoch_size = mnist.train.num_examples
# batch의 크기
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 15번의 epoch동안 반복
    for epoch in range(num_epochs):
        avg_cost = 0
        # 총 batch 수
        num_batch = int(epoch_size / batch_size)

        # 총 batch 수만큼 반복
        for i in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict = {x: batch_x,
                                                            y: batch_y})
            avg_cost += c / num_batch

        print("Epoch: ", "%04d" % (epoch+1), "Cost= ", "{:.9f}".format(avg_cost))

    print("\nAccuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                    y: mnist.test.labels}))

    # 예측 및 예측한 손글씨 출력
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(H, 1), feed_dict={x: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()