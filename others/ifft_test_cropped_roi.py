import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



img = cv2.imread('/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/2L_128_sImage_Image2_062.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(len(img[0])//2,len(img)//2))

rows, cols = img.shape
crow,ccol = rows//2 , cols//2
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

img2 = cv2.imread('/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/tmp/2L_128_sImage_Image2_063_cropped.jpg',cv2.IMREAD_GRAYSCALE)
# f = np.fft.fft2(img2)
f = cv2.dft(np.float32(img2),flags = cv2.DFT_COMPLEX_OUTPUT)
fshift_for_crop = np.fft.fftshift(f)
rows2, cols2 = img2.shape
print(rows2, cols2)

mask = 20*np.log(np.abs(fshift_for_crop)) < 290

# apply mask and inverse DFT
dft_shift[crow-rows2//2:crow+rows2//2+1, ccol-cols2//2:ccol+cols2//2+1] = dft_shift[crow-rows2//2:crow+rows2//2+1, ccol-cols2//2:ccol+cols2//2+1] * mask
f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()