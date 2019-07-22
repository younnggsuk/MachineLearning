# 9. Neural Net for XOR

## Source code

### xor_not_nn.py

- XOR 문제에 Neural Network을 적용 X
  - 학습이 제대로 되지 않음

### xor_nn.py

- XOR 문제에 Neural Network을 적용 O

  - 학습이 제대로 이루어짐

### xor_wide_nn.py

- XOR 문제에 Neural Network을 넓게 적용

  - weight, bias의 차원을 10으로 넓게 적용
  - 학습이 더 정확해짐

### xor_nn.py

- XOR 문제에 Neural Network을 넓고 깊게 적용 O
  - weight, bias의 차원을 10으로 더 넓게 적용
  - layer를 4개로 깊게 적용
  - 학습이 그냥 넓힌거보다 더 정확해짐

### ex_tensorboard.py

- tensorboard 사용 예제
  - tensorboard 사용 순서
    - log할 텐서 설정
      - scalar
        - tf.summary_scalar("Cost", cost)
      - non-scalar
        - tf.summary_histogram("Hypothesis", h)
    - summary 합치기
      - merged_summary = tf.summary.merge_all()
    - writer 생성 및 graph 추가
      - writer = tf.summary.FileWriter("./logs/log1")
      - writer.add_graph(sess.graph)
    - sess.run()으로 합친 summary 실행
      - summaryVal = sess.run(summary, feed_dict={x: xData, y: yData})
    - tensorboard 실행
      - \$ tensorboard --logidr=./logs/log1
