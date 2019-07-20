import tensorflow as tf

tf.InteractiveSession()

# 실수 정수로 변환
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())

# bool 정수로 변환
print(tf.cast([True, False], tf.int32).eval())