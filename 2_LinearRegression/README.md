# 2. Linear Regression

## Source code

### linear_regression.py

- Linear Regression
  - Data set : X = [1, 2, 3], Y = [1, 2, 3]
  - Hypothesis = Wx + b
  - Cost = (hypothesis - Y)의 제곱값들의 평균
  - tf.Variable 사용을 위해 tf.global_variable_initializer()
  - Gradient Descent Algorithm을 통해 cost를 minimize

### linear_regression_placeholder.py

- placeholder를 이용한 Linear Regression
  - 위의 예제와 같음
  - 학습시킨 hypothesis에 값 대입 후 결과 확인
