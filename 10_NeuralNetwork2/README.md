# 10. Neural Network2

## 개념

### Activation Function

- Sigmoid를 사용하면 Layer가 깊어지면 Vanishing Gradient 문제가 발생
- 따라서 Neural Network(Deep Learning)에서는 주로 ReLU를 activation function으로 사용

### Weight Initialization

- Weight의 초기값에 따라서도 Vanishing Gradient 문제가 발생
- Xavier Initialization를 사용하면 이를 해결할 수 있음

### Dropout, Ensemble

- Dropout
  - Neural Network(Deep Learning)에서 overfitting을 막기 위한 방법(Regularization)
  - 몇가지 뉴런들은 제외시킨 후 학습시키는 방식
  - 학습시에는 dropout을 적용, 실제 사용 시에는 모든 뉴런을 사용
    - Training : (dropout rate < 1)
    - Testing : (dropout rate == 1)
- Ensemble
  - 학습할 때마다 학습모델은 초기값이 같이 않으므로 서로 다름
  - 이러한 여러 학습 모델을 하나로 합치는 방법

## Source code

### Adam optimizer

### mnist_softmax.py

- 손글씨 학습 예제 1
  - Activation Function : Softmax
  - Weight Initialization : random 
  - Accuracy : 90%

### mnist_nn_relu.py

- 손글씨 학습 예제 2
  - Neural Network (Deep Learning) 적용
  - Activation Function : Relu
  - Weight Initialization : random
  - Layer : 3
  - Accuracy : 95%

### mnist_nn_xavier.py

- 손글씨 학습 예제 3
  - Neural Network (Deep Learning) 적용
  - Activation Function : Relu
  - Weight Initialization : Xavier
  - Layer : 3
  - Accuracy : 97%

### mnist_nn_xavier_deep.py

- 손글씨 학습 예제 4
  - Neural Network (Deep Learning) 적용
  - Activation Function : Relu
  - Weight Initialization : Xavier
  - Layer : 5
  - Accuracy : 97%
  - Layer를 더 깊게 하더라도 정확도가 큰 변화가 없음
  - 오히려 Overfitting이 더 일어날 수 있음

### mnist_nn_xavier_dropout.py

- 손글씨 학습 예제 5
  - Neural Network (Deep Learning) 적용
  - Dropout 적용
  - Activation Function : Relu
  - Weight Initialization : Xavier
  - Layer : 5
  - Accuracy : 98%
  - Dropout을 적용해 Overfitting을 방지, Accuracy가 더 올랐음

