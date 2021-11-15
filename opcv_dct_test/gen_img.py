import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(patH):
    blank_image = np.zeros((512,512,1), np.uint8)
    for i in range(16):
        if i%2 == 0:
            blank_image[128+16*i:128+16*(i+1),128:384] = 255
    blank_image[:,200] = 255
    # for i in range(16):
    #     if i%2 == 0:
    #         blank_image[40*i:40*(i+1)] = 255

    cv2.imwrite(patH+'stripe_cover_defect.png', blank_image)

    return 0

def remove_rb_sample(pathh):
    img1 = cv2.imread(pathh + 'stripe_but_largeImg_centered.png', cv2.IMREAD_GRAYSCALE)
    imgF = img1.astype('float')
    re = cv2.dct(imgF)
    img1 = cv2.imread(pathh+'stripe_cover_defect.png', cv2.IMREAD_GRAYSCALE)
    imgFloat = img1.astype('float')
    coeff = cv2.dct(imgFloat)
    reco = (coeff - re)
    reconsImg = cv2.idct(reco)*1000000
    plt.imshow(reconsImg,cmap='gray')
    plt.show()
    cv2.imwrite(pathh+'residual.png',reconsImg)

if __name__ == "__main__":
    # patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    # patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/opcv_dct_test/"
    # main(patH)
    remove_rb_sample(patH)