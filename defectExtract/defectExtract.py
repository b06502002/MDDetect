import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(pathh):
    ddepth = cv2.CV_64F
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    imgLst = []
    flagg = 0
    st = 288
    ed = 309
    for ii in range(st,ed):
        imgLst.append(cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE))
        if flagg == 0:
            Bimg = np.zeros_like(imgLst[0], dtype='float64')
            flagg = 1
        Bimg += imgLst[ii-st].astype('float64')

    Bimg //= (ed-st)
    plt.imshow(Bimg,cmap='gray')
    plt.show()

    for ii in range(1):
        diff = imgLst[ii]-Bimg
        plt.imshow(diff,cmap='gray')
        plt.show()
        diff_n = cv2.normalize(diff, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
        #cv2.imwrite('/home/cov/Desktop/PML/progress report/2021 Oct 1/'+'{}.jpg'.format(ii), diff_n)
        thresh1 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 25)# obtain white Mura
        thresh2 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51,-25)# obtain black Mura
        BothT = thresh1+thresh2
        BothT2 = cv2.medianBlur(BothT, 7)
        kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(BothT2, cv2.MORPH_OPEN, kernel)
        #cv2.imwrite('/home/cov/Desktop/PML/progress report/2021 Oct 1/'+'{}t.jpg'.format(ii), opening)
        plt.imshow(opening,cmap='gray')
        plt.show()

    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    main(patH)