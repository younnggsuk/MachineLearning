import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[0].reshape(28, 28)

sess = tf.InteractiveSession()

# Image : 28*28, 색 1개
img = img.reshape(-1, 28, 28, 1)
# Weight : 3*3, 색 1개, 필터 5개
w1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# Convolution
conv2d = tf.nn.conv2d(img, w1, strides=[1, 2, 2, 1], padding='SAME')
print(conv2d)

# Convolution 결과 출력
# sess.run(tf.global_variables_initializer())
# conv2d_img = conv2d.eval()
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14, 14), cmap='gray')
#
# plt.show()

# Pooling
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(pool)

# Pooling 결과 출력
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

plt.show()