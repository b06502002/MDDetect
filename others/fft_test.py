import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# img = cv.imread('/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/2L_128_sImage_Image2_062.jpg',cv.IMREAD_GRAYSCALE)
img = cv.imread('/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/tmp/2L_128_sImage_Image2_063_cropped.jpg',cv.IMREAD_GRAYSCALE)
# img = cv.imread('/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/opcv_dct_test/stripe_but_largeImg_centered.png',cv.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
# magnitude_spectrum = 20*np.abs(fshift)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()