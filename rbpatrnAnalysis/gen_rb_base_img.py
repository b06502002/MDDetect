import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(pathh):

    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    flagg = 0
    imgLst = []
    for ii in range(len(lst)):
        imgLst.append(cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE))
        if flagg == 0:
            Bimg = np.zeros_like(imgLst[0], dtype='float64')
            flagg = 1
        Bimg += imgLst[ii].astype('float64')

    Bimg //= len(lst)
    cv2.imwrite(pathh+'refimg1.png',Bimg)
    plt.imshow(Bimg,cmap='gray')
    plt.show()

if __name__ == "__main__":
    patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/ImgTmp/"
    main(patH)