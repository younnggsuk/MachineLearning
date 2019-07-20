import tensorflow as tf

tf.InteractiveSession()

x = [[1., 2.],
     [3., 4.]]

# 전체 평균
print(tf.reduce_mean(x).eval())
# 0, 1, 마지막(-1)축에 대해 평균
print(tf.reduce_mean(x, axis=0).eval())
print(tf.reduce_mean(x, axis=1).eval())
print(tf.reduce_mean(x, axis=-1).eval())

# 전체 합
print(tf.reduce_sum(x).eval())
# 0, 1, 마지막(-1)축에 대해 합
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=1).eval())
print(tf.reduce_sum(x, axis=-1).eval())
