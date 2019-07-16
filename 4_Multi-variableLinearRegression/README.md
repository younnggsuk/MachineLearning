# 4. Multi-variable Linear Regression

## Source code

### multi_variable_linear_regression.py

- Multi-variable Linear Regression 예제

  - H = XW + b ( X : 5 * 3 행렬, W : 3 * 1 행렬 )
  - H = x1w1 + x2w2 + x3w3 + b
  - 3개의 feature에 따른 결과 Data set을 Linear Regression에 적용
  - feature(x항)가 많아질 수록 코드가 복잡해짐

### multi_variable_matmul_linear_regression.py

- Multi-variable Linear Regression 예제 2

  - 앞의 예제의 feature(x항)를 matrix로 나타냄
  - tf.matmul()를 통해 코드를 간단하게 나타낼 수 있음
  - shape 잘 나타내는게 중요

### file_input_linear_regression.py

- File에서 Data를 읽어서 Multi-variable Linear Regression 적용
	
  - 하나의 File(*.csv)에서 데이터를 받아오는 예제
  - numpy의 loadtxt를 이용
  - 받아온 데이터를 slicing
  - Linear Regression 적용

### file_input_linear_regression2.py

- File에서 Data를 읽어서 Multi-variable Linear Regression 적용 2

  - 여러개의 File(*.csv)에서 데이터를 받아오는 예제
  - 데이터의 크기가 커서 메모리가 감당하지 못하는 경우 사용하는 Tensorflow의 Queue Runner를 이용
      - 데이터 파일을 Queue에 올림
      - 파일을 읽을 reader 설정
      - 데이터 parsing(decoding) 설정
      - batch를 통해 데이터 읽어들임
      - (batch를 통해 데이터를 섞어서 읽을 때는 suffle_batch 사용)
  - Linear Regression 적용