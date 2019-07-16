import tensorflow as tf

# 읽어들일 파일들
filenames = []
for i in range(4):
    filenames.append("test-score%d.csv" % (i+1))

# 위의 파일들을 queue에 올림
filename_queue = tf.train.string_input_producer(
    filenames, shuffle=False, name='filename_queue'
)

# 파일을 읽을 Reader설정
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 파일에서 읽어들인 값 parsing(decoding) 설정
record_defaults = [[0.], [0.], [0.], [0.]]
data = tf.decode_csv(value, record_defaults=record_defaults)

# 데이터를 읽어올 수 있게 해주는 일종의 펌프 역할을 하는 batch 생성
# batch를 섞어서 사용할 때는 tf.train.shuffle_batch사용
train_x_batch, train_y_batch = \
    tf.train.batch([data[0:-1], data[-1:]], batch_size=25)


x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(H-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Queue Runner에서 사용할 쓰레드를 관리하는 Coordinator생성
# Queue Runner에서 사용할 쓰레드 생성
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, H_val, _ = sess.run(
        [cost, H, train], feed_dict={x: x_batch, y: y_batch}
    )

    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", H_val)

# Queue Runner 정지 및 쓰레드 정지
coord.request_stop()
coord.join(threads)