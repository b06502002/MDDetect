import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    H = np.zeros((P, Q))

    for u in range(0, P):
        for v in range(0, Q):
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0
    return H

def readhsv(patH):
    frame = cv.imread(patH, cv.COLOR_BGR2HSV)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # Note that converting to HSV decreases the image size
    frame_threshold = cv.inRange(frame_HSV, (0, 24, 50), (100, 70, 255)) # (H,S,V)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gen_image = frame_gray * (frame_threshold//255) + 0 * (1-(frame_threshold//255))

    return frame_gray, gen_image

def main(patH):
    ori_img, res_img = readhsv(patH)

    f = np.fft.fft2(ori_img)
    fshift = np.fft.fftshift(f)
    phase_spectrumR = np.angle(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    f_2 = np.fft.fft2(res_img)
    fshift_2 = np.fft.fftshift(f_2)
    phase_spectrumR_2 = np.angle(fshift_2)
    magnitude_spectrum_2 = 20*np.log(np.abs(fshift_2))


    img_shape = ori_img.shape
    NotchFilter = notch_reject_filter(img_shape, 60, 0,0)
    RegionReject = np.fft.ifftshift(fshift-fshift_2*(1-NotchFilter)*1)
    inverse_RegionReject = np.fft.ifft2(RegionReject)  # Compute the inverse DFT of the result


    Result = np.abs(inverse_RegionReject)
# -----
    plt.subplot(221)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude spectrum')

    plt.subplot(222)
    plt.imshow(ori_img, cmap='gray')
    plt.title('Original')

    plt.subplot(223)
    plt.imshow(magnitude_spectrum_2*(1-NotchFilter), "gray")
    plt.title("pattern spectrum")

    plt.subplot(224)
    plt.imshow(Result, "gray") 
    plt.title("Result")

    plt.show()

# -----
    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/4L_48_sImage_Image4_074.jpg" 
    main(patH)
