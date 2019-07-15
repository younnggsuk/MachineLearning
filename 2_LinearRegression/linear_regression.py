import tensorflow as tf

# training data set
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Weight, bias
# 0~1 사이의 정규확률분포 값을 가지는 변수, shape : [1]
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = Wx + b
hypothesis = W * x_train + b

# cost function = (hypothesis - y)^2의 평균
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Gradient Descent 알고리즘을 이용해 cost 함수를 최소화하는 train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 그래프 생성
sess = tf.Session()

# 모든 변수 초기화
sess.run(tf.global_variables_initializer())

# 2001번 학습 수행, 20번마다 cost, Weight, bias 출력
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(sess.run(cost), sess.run(W), sess.run(b))