import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(patH):
    img1 = cv2.imread(patH+'stripe'+'.png',cv2.IMREAD_GRAYSCALE)
    imgFloat = img1.astype('float')
    coeff = cv2.dct(imgFloat)
    plt.imshow(coeff[0:50,0:50],cmap='gray')
    plt.show()
    cv2.imwrite(patH+'stripe_coeff.png',coeff)
    coeff[:,0] = 0
    reconsImg = cv2.idct(coeff)
    # plt.imshow(reconsImg,cmap='gray')
    # plt.show()
    return 0

if __name__ == "__main__":
    # patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    # patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/opcv_dct_test/"
    main(patH)