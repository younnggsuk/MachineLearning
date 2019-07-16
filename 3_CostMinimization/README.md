# 3. Cost Minimization

## Source code

### cost_minimization_show_graph.py

- Cost Minimation 예제

  - 단순히 Weight를 변화시키며 Cost를 관찰
  - Weight, Cost의 변화를 그래프로 출력

### gradient_descent.py

- Gradient Descent Algorithm 예제

  - Gradient에 직접 구한 식(미분한 식)을 적용
  - 식을 통해 얻어진 값으로 Weight를 직접 증가/감소

### gradient_descent_optimizer.py

- Gradient Descent Algorithm 예제 2

  - Gradient에 tensorflow에서 제공하는 optimizer를 통해 얻은 식 적용
  - 식을 통해 얻어진 값으로 Weight를 직접 증가/감소

### gradient_descent_compare.py

- Gradient Descent Algorithm 예제 3

  - 직접 구한 gradient와 Tensorflow의 optimizer를 통한 gradient를 비교
  - 큰 차이가 없음
  - optimizer.compute_gradients()를 통해 매 순간의 gradient를 확인 가능
  - optimizer.apply_gradients()를 통해 Gradient를 직접 적용할 수 있음
