import tensorflow as tf

tf.InteractiveSession()

# shape: [1, 2]
m1 = tf.constant([[3., 3.]])
# shape: [2, 1]
m2 = tf.constant([[2.], [2.]])

# broadcasting
# 연산을 수행할 수 있도록 shape가 [2, 2]가 되도록 늘어남
print((m1+m2).eval())


# shape: [1, 2]
m3 = tf.constant([[3., 3.]])
# shape: [2, 1]
m4 = tf.constant([[2.], [2.]])

# 행렬 곱셈
print(tf.matmul(m3, m4).eval())

# 행렬 곱셈 X
# broadcasting 후 행렬의 원소 끼리 곱셈이 일어남
print((m3*m4).eval())

