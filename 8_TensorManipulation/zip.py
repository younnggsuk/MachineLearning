# 배열에서 첫번째 원소들을 x, y에 하나씩 넣음
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

# 배열에서 첫번째 원소들을 x, y, z에 하나씩 넣음
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)