import tensorflow as tf
import numpy as np

# XOR data set
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Logistic Regression
h = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

training = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(training, feed_dict={x: x_data, y: y_data})

    if step % 200 == 0:
        c_val = sess.run(cost, feed_dict={x: x_data, y: y_data})
        w_val = sess.run(w)
        print(step, c_val, w_val)


h_val, prediction_val, accuracy_val = sess.run([h, prediction, accuracy],
                                               feed_dict={x: x_data})
print("\nHypothesis\n{}\nPrediction\n{}\nAccuracy\n{}".format(h_val, prediction_val, accuracy_val))
