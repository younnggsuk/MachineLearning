import tensorflow as tf
import numpy as np

tf.InteractiveSession()

# 2*2 image
image = np.array([[[[4], [3]],
                   [[2], [1]]]], np.float32)

# Max pooling
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')

print(pool.shape)
print(pool.eval())
