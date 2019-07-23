import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 100
num_epochs = 15
num_iterations = int(mnist.train.num_examples / batch_size)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Deep Layer
w1 = tf.get_variable("weight1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.get_variable("weight2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, w2) + b2)

w3 = tf.get_variable("weight3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, w3) + b3)


w4 = tf.get_variable("weight4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, w4) + b4)

w5 = tf.get_variable("weight5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))

h = tf.matmul(L4, w5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))
training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.equal(tf.argmax(h, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epochs):
    avg_cost = 0

    for iteration in range(num_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([training, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += (cost_val / num_iterations)

    print(f"Epoch: {(epoch+1):04d}, Cost:{avg_cost:.9f}")

print("Learning Finished!")

print(
    "Accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
)

r = random.randint(0, mnist.test.num_examples-1)

print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], axis=1)))
print(
    "Prediction:",
    sess.run(tf.argmax(h, axis=1), feed_dict={x:mnist.test.images[r:r+1]})
)

plt.imshow(
    mnist.test.images[r:r+1].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest"
)
plt.show()