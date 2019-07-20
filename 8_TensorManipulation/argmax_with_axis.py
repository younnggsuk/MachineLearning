import tensorflow as tf

tf.InteractiveSession()

x = [[0, 1, 2],
     [2, 1, 0]]

# 0, 1, 마지막(-1)축에 대해 가장 큰 원소의 index
print(tf.argmax(x, axis=0).eval())
print(tf.argmax(x, axis=1).eval())
print(tf.argmax(x, axis=-1).eval())
