# 4. Multi-variable Linear Regression

## Source code

### multi_variable_linear_regression.py

- Multi-variable Linear Regression 예제

  - H = XW + b ( X : 5 _ 3 행렬, W : 3 _ 1 행렬 )
  - H = x1*w1 + x2*w2 + x3\*w3 + b
  - 3개의 feature에 따른 결과 Data set을 Linear Regression에 적용
  - feature(x항)가 많아질 수록 코드가 복잡해짐

### multi_variable_matmul_linear_regression.py

- Multi-variable Linear Regression 예제 2

  - 앞의 예제의 feature(x항)를 matrix로 나타냄
  - tf.matmul()를 통해 코드를 간단하게 나타낼 수 있음
  - shape 잘 나타내는게 중요
