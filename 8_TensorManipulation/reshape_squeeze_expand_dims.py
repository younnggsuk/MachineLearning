import tensorflow as tf

tf.InteractiveSession()

t1 = tf.constant(
    [
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [6, 7, 8],
            [9, 10, 11]
        ]
    ]
)
# 초기 shape: [2, 2, 3]
print(tf.shape(t1).eval())

# [?, 3] 형태로 reshape
t2 = tf.reshape(t1, shape=[-1, 3])
# shape: [4, 3]
print(tf.shape(t2).eval())

# [?, 1, 3] 형태로 reshape
t3 = tf.reshape(t1, shape=[-1, 1, 3])
# shape: [4, 1, 3]
print(tf.shape(t3).eval())

# 차원 중 사이즈가 1인 것을 찾아 스칼라로 바꿈
print(tf.squeeze([[0], [1], [2]]).eval())
# 따라서 아래의 경우 사이즈가 1인 것이 없으므로 변화 없음
print(tf.squeeze([[0, 1], [1, 2], [2, 3]]).eval())


# 1축에 대해 차원 추가
print(tf.expand_dims([0, 1, 2], axis=1).eval())
# 위의 결과의 1축에 차원 추가
print(tf.expand_dims(tf.expand_dims([0, 1, 2], axis=1), axis=1).eval())
