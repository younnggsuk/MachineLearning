import tensorflow as tf

tf.InteractiveSession()

x = [1, 4]
y = [2, 5]
z = [3, 6]

# 그냥 순서대로 쌓음
print(tf.stack([x, y, z]).eval())

# 축1에 대해 쌓음
print(tf.stack([x, y, z], axis=1).eval())

