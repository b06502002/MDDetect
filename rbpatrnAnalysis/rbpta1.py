import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(pathh): # read reference background and obtain DCT coefficients

    img1 = cv2.imread(pathh + "refimg1.png", cv2.IMREAD_GRAYSCALE)
    imgFloat = img1.astype('float')
    coeff = cv2.dct(imgFloat)
    return coeff

def main2(pathh):    # for old path
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    imgLst = []
    flagg = 0
    st = 288
    ed = 301
    for ii in range(st,ed):
        imgLst.append(cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE))
        if flagg == 0:
            Bimg = np.zeros_like(imgLst[0], dtype='float64')
            flagg = 1
        Bimg += imgLst[ii-st].astype('float64')
    Bimg //= (ed-st)
    imgFloat = Bimg.astype('float')
    return cv2.dct(imgFloat)

def remove_rb(pathh, re): # remove rainbow patterns 
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    plt.imshow(cv2.idct(re),cmap='gray')
    plt.show()
    for ii in range(288,301):#(11):
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        imgFloat = img1.astype('float')
        coeff = cv2.dct(imgFloat)
        reco = coeff - re
        reconsImg = cv2.idct(reco)
        plt.imshow(reconsImg,cmap='gray')
        plt.show()

if __name__ == "__main__":
    # patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/ImgTmp/"
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    re = main2(patH)
    remove_rb(patH, re)