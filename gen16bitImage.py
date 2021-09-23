import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1,img8bit = np.zeros((256,256)), np.zeros((256,256))
for i in range(256):
    for j in range(256):
        img1[j,i] = i*256+j
    img8bit[:,i] = i
# img_scaled = cv2.normalize(img1, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
# plt.imshow(img_scaled,cmap='gray',vmin=0, vmax=65535)
# plt.show()
# print(img1[:,255])
# cv2.imwrite("16bit.png",img1.astype(np.uint16)); cv2.imwrite("8bit.png",img8bit)
img2 = img1.astype(np.uint8)
print(img2[:,255])
img3 = img1.astype(np.uint16)
print(img3[:,255])
cv2.imwrite("1.png",img1), cv2.imwrite("2.png",img1.astype(np.uint8))

