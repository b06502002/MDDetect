import cv2
import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

img = cv2.imread('/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/2L_128_sImage_Image2_063.jpg', 0)
# img = cv2.imread('/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/tmp/2L_128_sImage_Image2_063_cropped.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20*np.log(np.abs(fshift))

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 3,8,1) # cropped
H2 = notch_reject_filter(img_shape, 5,-15,5) # cropped
# H1 = notch_reject_filter(img_shape, 15, 0, 0)
# H2 = notch_reject_filter(img_shape, 10, 0, 0)
# H_compl = notch_reject_filter(img_shape, 2, 13, 0)
# H_compl2 = notch_reject_filter(img_shape, 2, 0, 13)
# H3 = notch_reject_filter(img_shape, 4, -12, 8)
# H4 = notch_reject_filter(img_shape, 4, -30, 16)
# H5 = notch_reject_filter(img_shape, 3, 15, 7)
# H6 = notch_reject_filter(img_shape, 3, 6, 2)

# NotchFilter = (1 - (H2-H1) + (1-H_compl*H_compl2))*H3*H4*H5*H6
NotchFilter = H1*H2
NotchRejectCenter = fshift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result


Result = np.abs(inverse_NotchReject)

plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(221)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum*NotchFilter, "gray") 
plt.title("Notch Reject Filter")

plt.subplot(224)
plt.imshow(Result, "gray") 
plt.title("Result")


plt.show()