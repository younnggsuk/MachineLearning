# 6. Multinomial Logistic Regression

## Source code

### softmax_classifier.py

- Multinomial Logistic Regression 예제
  - Binary가 아닌 경우의 Classification
  - Softmax 함수를 적용한 Hypothesis를 사용
  - Cross-Entropy를 적용한 Cost Function을 사용
  - Gradient Descent Algorithm
  - tf.arg_max()를 통해 Hypothesis의 결과 값중 가장 큰 값을 해당 Class로 간주

### fancy_softmax_classification.py

- Multinomial Logistic Regression 예제 2
  - 위의 예제에서 함수를 직접 입력한 부분을 Tensorflow의 함수로 대체
  - File에서 Data set 받아옴
  - tf.one_hot(), tf.reshape()를 사용해 y data를 one_hot 형태로 변환
  - tf.nn.softmax_cross_entropy_with_logits를 이용해 Cross-Entropy 적용
  - 학습 후 예측 값과 실제 값의 비교 출력
