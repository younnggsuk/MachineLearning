# 7. Practice & Some Tips

## 개념

### Training data & Test data

- Training data로 평가하면 의미가 없음
- 따라서 Training data와 Test data는 따로 분리되어야 함
- Training data만으로 학습시키고 Test data만으로만 평가해야함

### Learning Rate

- Learning rate(alpha값)은 너무 크거나 너무 작으면 안됨
- Learning rate이 너무 크면 Overshooting이 일어날 수 있음
- Learning rate가 너무 작으면 시간이 오래 걸리거나 local minimum에 갇힐 수 있음
- 다양한 Learning rate을 사용해 보는 것이 최선

### Data Preprocessing

- 데이터의 범위가 너무 크면 제대로 학습시키기가 어려움(gradient descent algorithm에서 최저점으로 가기 어려움)
- 따라서, 데이터의 범위를 줄여주는(re-scaling) Preprocessing이 필요함
- Preprocessing 방법에는 Normalization(정규화), Standardization(표준화)이 있음
- Normalization (MinMax)

  - x = x-xmin / xmax-xmin
  - 최소값을 빼준 후 전체 범위로 나눔
  - 0~1범위로 축소 가능

- Standardization (Z-score normalization)

  - x = x-xmean / xstd
  - 평균을 빼준 후 표준편차로 나눔

### Overfitting

- Data를 과하게 학습시켰다는 의미이며 Testing data에만 잘 학습이 된 경우 발생
- 따라서 실제 데이터 적용 시 제대로 예측이 되지 않음
- 해결방법
  - 더 많은 학습 데이터로 학습 시키기
  - feature의 수를 줄이기
  - Regularization(일반화)
    - 너무 fitting되지 않도록 weight를 너무 작지 않게 하는 방법
    - cost 함수에 regularization strength라는 항을 추가하고
    - regularization strength 항을 적절히 조절하는 방식

### Epoch, Batch, Iteration

- 머신러닝에서는 일반적으로 여러번의 학습과정을 거침
- 메모리의한계와 속도 저하 때문에 여러번에 걸쳐 데이터를 나누어 주게됨
- 이때 사용되는 용어가 Epoch, Batch, Iteration이 있음
- Epoch
  - 전체 데이터 셋을 한번 학습
- batch
  - 전체 데이터(Epoch)을 나눈 것
- Iteration
  - 전체 데이터를 몇번 나누어 주는가
- 따라서, 총 데이터가 100개이고, 20개씩 나누어 준다면,
- Epoch : 100, Batch : 20, Iteration : 5이다.

## Source code

### training_and_datasets.py

- Training & Test data
  - training data와 test data를 분리
  - training data만으로 학습
  - test data만으로 평가

### big_learning_rate.py

- Learning Rate가 큰 경우
  - Overshooting이 일어나 Cost와 Weight가 nan(무한대)로 나타남

### small_learning_rate.py

- Learning Rate가 작은 경우
  - Cost값이 너무 조금씩 줄어들어서 충분히 작아지지 못해 제대로 학습이 되지 않음

### linear_regression_without_minmax.py

- Preprocessing 예제
  - 데이터의 범위가 너무 커서 제대로 학습이 되지 않음

### linear_regression_minmax.py

- Preprocessing 예제 2
  - 위의 linear_regression_without_minmax에서의 데이터에 normalization을 적용
  - minmax를 이용해 0~1 데이터로 전처리 후 적용

### mnist.py

- 손글씨 학습 예제
  - tensorflow.examples.tutorials.mnist에 있는 손글씨 학습 데이터 예제
  - 0~9까지의 숫자 손글씨를 학습 데이터를 통해 학습하고 잘 되었는지 출력
