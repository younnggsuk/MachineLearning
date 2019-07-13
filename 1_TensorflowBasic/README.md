# 1. Tensorflow Basic

## Source code

### hello_tensorflow.py

- Tensorflow 기본 예제
  - Hello Tensorflow! 출력

### computational_graph.py

- 계산 그래프 예제
  - 상수와 연산을 node로 나타내고 session을 통해 수행
  - 상수는 constant 사용

### placeholder.py

- 계산 그래프 예제
  - 변수와 연산을 node로 나타내고 session을 통해 수행
  - 변수는 placeholder, feed_dict 사용

## 개념

### Rank

- 배열의 차원
- Scalar (Rank 0)
  - s = 483
- Vector (Rank 1)
  - v = [1.1, 2.2, 3.3]
- Matrix (Rank 2)
  - m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
- 3-Tensor (Rank 3)
  - t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]
- n-Tensor (Rank n)
  - .....

### Shape

- 배열의 모양
- [D0, D1, ..., Dn-1]와 같이 나타내며 D0, D1들은 각 차원에서의 원소 수를 나타냄
- Scalar (Rank 0, shape [])
  - 3
- Vector (Rank 1, shape [3])
  - [1., 2., 3.]
- Matrix (Rank 2, shape [2, 3])
  - [[1., 2., 3.], [4., 5., 6.]]
- 3-Tensor (Rank 3, shape [2, 1, 3])
  - [[[1., 2., 3.]], [[7., 8., 9.]]]

### Type

- DT_FLOAT
  - 32bit, floating point
  - tf.float32
- DT_DOUBLE
  - 64bit, floating point
  - tf.float64
- DT_INT8
  - 8bit, signed integer
  - tf.int8
- DT_INT16
  - 16bit, signed integer
  - tf.int16
- DT_INT32
  - 32bit, signed integer
  - tf.int32
- DT_INT64
  - 64bit, signed integer
  - tf.int64
