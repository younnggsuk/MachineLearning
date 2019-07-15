import tensorflow as tf
import matplotlib.pyplot as plt

# Data set
X = [1, 2, 3]
Y = [1, 2, 3]

# Weight 값 변경시키면서 결과 보기 위해 placeholder
W = tf.placeholder(tf.float32)

# H = Wx
hypothesis = W * X

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


W_val = []
cost_val = []

# W를 -3 ~ +5까지 변화시키며 W, cost를 리스트에 담음
for i in range(-30, 51):
    feed_W = i*0.1
    cur_W, cur_cost = sess.run([W, cost], feed_dict={W:feed_W})
    W_val.append(cur_W)
    cost_val.append(cur_cost)


# 그래프 출력
plt.plot(W_val, cost_val)
plt.show()
