import numpy as np

# 1차원 배열 생성
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 배열 출력
print(t)
# shape, rank 출력
print(t.shape, t.ndim)
# 1번째, 2번째, 마지막 원소 출력
print(t[0], t[1], t[-1])
# index 2~4, 4~5 출력
print(t[2:5], t[4:6]) # print(t[2:5], t[4:-1])
# index 처음~2, 3~끝 출력
print(t[:2], t[3:])