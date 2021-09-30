import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile.tifffile import astype

def main(pathh):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    # ct = 0
    # for fname in os.listdir(pathh):
    #     img1 = cv2.imread(pathh + fname)
    #     print(fname)
    #     imgFloat = img1.astype('uint8')
    #     print(imgFloat.max())
    #     plt.imshow(imgFloat)
    #     plt.show()
    #     ct += 1
    #     if ct == 3:
    #         break
    imgLst = []
    for ii in range(3):
        imgLst.append(cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE))
        if ii == 0:
            Bimg = np.zeros_like(imgLst[0], dtype='float64')
        Bimg += imgLst[ii].astype('float64')

    Bimg //= 3
    plt.imshow(Bimg,cmap='gray')
    plt.show()

    for ii in range(3):

        plt.imshow(imgLst[ii]-Bimg,cmap='gray')
        plt.show()

    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    main(patH)