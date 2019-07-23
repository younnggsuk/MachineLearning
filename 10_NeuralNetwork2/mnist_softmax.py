import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

learning_rate = 0.001
batch_size = 100
num_epochs = 50
num_iterations = int(mnist.train.num_examples / batch_size)

h = tf.matmul(x, w) + b
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