import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.InteractiveSession()

# 1장, 3*3, 1색
# shape : [1, 3, 3, 1]
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], np.float32)
print("image.shape: ", image.shape)

# 2*2, 1색, 3필터
# shape : [2, 2, 1, 3]
weight = np.array([[[[1, 10, -1]], [[1, 10, -1]]],
                   [[[1, 10, -1]], [[1, 10, -1]]]], np.float32)
print("weight.shape: ", weight.shape)

# convolution
# stride : 1*1
# padding : SAME이므로 convolution의 결과는 원본 이미지 크기와 같음
# 필터가 3개였으므로, 3장의 이미지가 나옴
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d.shape: ", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(3, 3), cmap='gray')

plt.show()