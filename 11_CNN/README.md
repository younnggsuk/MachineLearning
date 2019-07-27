# 11. CNN

## Source code

### cnn_ex.py

- CNN 예제 1
  - Convolution 예제

### cnn_ex_padding_same.py

- CNN 예제 2
  - Convolution 예제
  - convolution에서 padding을 'SAME'으로 주면 출력 이미지의 크기가 입력 이미지의 크기와 같음

### cnn_ex_3filter.py

- CNN 예제 3
  - Convolution 예제
  - Weight의 필터를 3개 적용

### cnn_ex_pooling.py

- CNN 예제 4
  - Pooling 예제

### cnn_ex_mnist.py

- CNN 예제 5
  - Mnist 사진 1장에 Convolution, Pooling 적용

### mnist_cnn.py

- 손글씨 학습 예제 1
  - Mnist에 CNN 적용
  - Convolutional Layer1
  	- 3*3, 32 channels
	- relu
	- pooling : 28*28 -> 14*14
  - Convolutional Layer2
  	- 3*3, 64 channels
	- relu
	- pooling : 14*14 -> 7*7
  - Fully-connected Layer3
	- 7*7*64 -> 10
  - Accuracy : 0.98

### mnist_cnn_deep_dropout.py

- 손글씨 학습 예제 2
  - Mnist에 더 깊은 CNN, Dropout 적용
  - Convolutional Layer1
  	- 3*3, 32 channels
	- relu
	- pooling : 28*28 -> 14*14
	- dropout : 0.7
  - Convolutional Layer2
  	- 3*3, 64 channels
	- relu
	- pooling : 14*14 -> 7*7
	- dropout : 0.7
  - Convolutional Layer3
  	- 3*3, 128 channels
	- relu
	- pooling : 7*7 -> 4*4
	- dropout : 0.7
  - Fully-connected Layer4
	- relu
	- 4*4*128 -> 625
	- dropout : 0.7
  - Fully-connected Layer5
  	- 625 -> 10
  - Accuracy : 0.99

### mnist_class_layers_ensemble.py

- 손글씨 학습 예제 3
  - Mnist에 더 깊은 CNN, Dropout 적용 (앞의 예제와 동일)
  - class와 tf.layers의 함수들을 사용해 코드를 더 간결하게 함
  - 2가지 모델을 학습시키고 예측값을 더하는 ensemble 적용
