import tensorflow as tf

tf.InteractiveSession()

t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
# 깊이가 3인 one_hot 생성 (차원이 1추가됨)
print(t)
# reshape로 차원 1 줄임
print(tf.reshape(t, shape=[-1, 3]).eval())
