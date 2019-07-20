import tensorflow as tf

tf.InteractiveSession()

x = [[0, 1, 2],
     [2, 1, 0]]

# 위 행렬과 같은 모양의 1로 된 배열 생성
print(tf.ones_like(x).eval())

# 위 행렬과 같은 모양의 0으로 된 배열 생성
print(tf.zeros_like(x).eval())
