import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 15
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        # 각 모델의 name마다의 namespace에 데이터 저장 (모델마다 각자 다른 데이터)
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
            self.y = tf.placeholder(tf.float32, shape=[None, 10])
            x_img = tf.reshape(self.x, [-1, 28, 28, 1])

            # Convolution layer1
            conv1 = tf.layers.conv2d(inputs=x_img, filters=32, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # Convolution layer2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # Convolution layer3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            # Fully-connected layer4
            flat = tf.reshape(dropout3, [-1, 4*4*128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # Fully-connected layer5
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # 각 모델의 namespace에 상관 없는 데이터 저장 (모델에 공통인 데이터)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        is_correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.x: x_test,
                                                     self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test,
                                                       self.y: y_test,
                                                       self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data,
                                                                     self.y: y_data,
                                                                     self.training: training})


sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model"+str(m+1)))

sess.run(tf.global_variables_initializer())

print("Learning Start!")

for epoch in range(epochs):
    avg_cost_list = np.zeros(len(models))
    batches = int(mnist.train.num_examples / batch_size)

    for i in range(batches):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        for model_idx, model in enumerate(models):
            c, _ = model.train(batch_xs, batch_ys)
            avg_cost_list[model_idx] += (c / batches)

    print("Epoch: %04d" % (epoch+1), "Cost: ", avg_cost_list)

print("Learning Finish!")

test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])

for model_idx, model in enumerate(models):
    print(model_idx, "Accuracy: ", model.get_accuracy(mnist.test.images,
                                                      mnist.test.labels))

    # 각 모델들의 예측을 다 더하는 방식의 Ensemble
    p = model.predict(mnist.test.images)
    predictions += p

# Ensemble
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1),
                                       tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

print("Ensemble Accuracy: ", sess.run(ensemble_accuracy))
