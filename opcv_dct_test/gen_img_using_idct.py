import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(patH):
    b_img = np.zeros((256,256,1), np.float)
    coeff = b_img #.astype('float')
    coeff[4][0] = 200000
    coeff2 = coeff.astype('float')
    reconsImg = cv2.idct(coeff2)
    cv2.imwrite(patH+'reconstructed_C4R0is200000_256.png', reconsImg)
    plt.imshow(reconsImg,cmap='gray')
    plt.show()
    return 0

if __name__ == "__main__":
    # patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    # patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/opcv_dct_test/"
    main(patH)