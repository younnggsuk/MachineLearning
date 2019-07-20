import tensorflow as tf

tf.InteractiveSession()

# rank 1, shape [4]
t1 = tf.constant([1, 2, 3, 4])
# rank 2, shape [2, 2]
t2 = tf.constant([[1, 2], [3, 4]])
# rank 4, shape [1, 2, 3, 4]
t3 = tf.constant(
    [# axis 0
        [# axis 1
            [# axis 2
                [# axis 3(-1)
                    1, 2, 3, 4
                ],
                [
                    5, 6, 7, 8
                ],
                [
                    9, 10, 11, 12
                ]
            ],
            [
                [
                    13, 14, 15, 16
                ],
                [
                    17, 18, 19, 20
                ],
                [
                    21, 22, 23, 24
                ]
            ]
        ]
    ]
)

# shape 출력
print(tf.shape(t1).eval())
print(tf.shape(t2).eval())
print(tf.shape(t3).eval())