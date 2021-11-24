import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/2L_128_sImage_Image2_062.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(len(img[0])//2,len(img)//2))

rows, cols = img.shape
# crow,ccol = rows//2 , cols//2
print(rows//2,cols//2)
crow = int(input("\nplease input coordinate of row:   \n"))
ccol = int(input("\nplease input coordinate of column:\n"))

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-15:crow+15, ccol-15:ccol+15] = 1
mask = 1-mask

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()