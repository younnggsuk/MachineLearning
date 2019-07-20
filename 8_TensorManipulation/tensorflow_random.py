import tensorflow as tf

tf.InteractiveSession()

print(tf.random_normal([3]).eval())
print(tf.random_uniform([3]).eval())
print(tf.random_uniform([2, 3]).eval())