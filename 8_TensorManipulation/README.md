# 8. Tensor Manipulation

## Source code

### simple_array.py

- 1차원 행렬 shape, rank 출력, slicing 예제

### 2d_array.py

- 2차원 헹렬 shape, rank 출력 예제

### shape_rank_axis.py

- shape, rank, axis 예제
  - axis 설명 주석 참고

### matmul_and_broadcasting.py

- 행렬 곱셈, broadcasting 예제
  - 행렬의 차원이 다를 때 같게 맞춰주는걸 broadcasting이라 함
  - 행렬의 곱셈은 matmul로 해야함
  - 그냥 곱하기 하면 broadcasting 일어나고 원소간의 연산이 일어남

### tensorflow_random.py

- tensorflow의 난수 생성 함수 예제
  - random_normal()
    - 정규 분포 난수
  - random_uniform()
    - 균등 분포 난수

### reduce_mean_and_sum.py

- reduce_mean(), reduce_sum() 예제
  - reduce_mean()
    - 평균
  - reduce_mean()
    - 합

### argmax_with_axis.py

- argmax() 예제
  - 가장 큰 원소의 index 알려줌
  - axis에 대해서도 가능

### reshape_squeeze_expand_dims.py

- reshape(), squeeze(), expand_dims() 예제
  - reshape()
    - 행렬 형태 변환 함수
    - -1은 알아서 해라는 뜻
    - 주로 마지막 원소의 수만 정하고 알아서 하도록 [-1, 원소 수] 형태로 씀
    - 주석 참고
  - squeeze()
    - 차원 중 사이즈가 1인 것을 찾아 스칼라로 변환해주는 함수
    - 1인게 없으면 변화 X
  - expand_dims()
    - 차원 추가 함수
    - axis에 대해서도 가능

### one_hot.py

- one_hot() 예제
  - one_hot형태로 바꿔주는 함수
  - 차원이 1 추가되므로 reshape로 다시 하나 낮춰주는 형태로 주로 사용

### casting.py

- casting() 예제
  - type casting 함수

### stack.py

- stack() 예제
  - 여러 행렬 쌓는 함수
  - axis에 대해서도 가능

### ones_like_zeros_like.py

- ones_like(), zeros_like() 예제
  - ones_like()
    - 모든 원소가 1로 된 배열 생성
  - zeros_like()
    - 모든 원소가 0으로 된 배열 생성

### zip.py

- zip() 예제
  - 여러 배열의 원소 순서대로 꺼내서 묶어주는 함수

### 그 외 transpose(전치함수) 같은 것들도 있음

### 그때 그때 찾아서 사용
