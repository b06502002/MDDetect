import os
import cv2
import math
import numpy as np
from numpy.lib.function_base import _i0_1
import pandas as pd
import matplotlib.pyplot as plt

def nothing(x):
    pass

def DCTprocess(IMG,Rlb,Rub,Clb,Cub,Clb2,Cub2):
    imgFloat = IMG.astype('float')
    coeff = cv2.dct(imgFloat)
    coeff[:,Rlb:Rub]=0
    coeff[Clb:Cub][:]=0
    coeff[Clb2:Cub2][:]=0
    reconsImg = cv2.idct(coeff)
    reconsImgforCV = reconsImg - reconsImg.min()
    reconsImgforCV /= reconsImgforCV.max()
    reconsImgforCV *= 255
    reconsImgforCV = reconsImgforCV.astype('uint8')
    return reconsImgforCV

def CropImg(IMG):
    canny_output = cv2.Canny(IMG, 10, 255)
    
    _, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (100,100,255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)
    return 0

def thres(IMG,adp_c,adp_m,med_k):
    diff_n = cv2.normalize(IMG, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    thresh1 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adp_m, adp_c)# obtain white Mura
    thresh2 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adp_m,-adp_c)# obtain black Mura
    BothT = thresh1+thresh2
    BothT2 = cv2.medianBlur(BothT, med_k)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(BothT2, cv2.MORPH_OPEN, kernel)
    return opening

def main(pathh,pathtoimg):

    cv2.namedWindow('trackbar')
    cv2.createTrackbar('Img1', 'trackbar', 0, 310, nothing)
    cv2.createTrackbar('Img2', 'trackbar', 0, 310, nothing)
    cv2.createTrackbar('Img3', 'trackbar', 0, 310, nothing)
    cv2.createTrackbar('dctCCoeff_lb', 'trackbar', 0, 1000, nothing)
    cv2.createTrackbar('dctCCoeff_ub', 'trackbar', 0, 1000, nothing)
    cv2.createTrackbar('dctCCoeff_lb2', 'trackbar', 0, 1000, nothing)
    cv2.createTrackbar('dctCCoeff_ub2', 'trackbar', 0, 1000, nothing)
    cv2.createTrackbar('dctRCoeff_lb', 'trackbar', 0, 60, nothing)
    cv2.createTrackbar('dctRCoeff_ub', 'trackbar', 0, 60, nothing)
    
    while(1):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        
        Clb = cv2.getTrackbarPos('dctCCoeff_lb', 'trackbar')
        Cub = cv2.getTrackbarPos('dctCCoeff_ub', 'trackbar')
        Clb2 = cv2.getTrackbarPos('dctCCoeff_lb2', 'trackbar')
        Cub2 = cv2.getTrackbarPos('dctCCoeff_ub2', 'trackbar')
        Rlb = cv2.getTrackbarPos('dctRCoeff_lb', 'trackbar')
        Rub = cv2.getTrackbarPos('dctRCoeff_ub', 'trackbar')
        i1 = cv2.getTrackbarPos('Img1', 'trackbar')
        i2 = cv2.getTrackbarPos('Img2', 'trackbar')
        i3 = cv2.getTrackbarPos('Img3', 'trackbar')

        img1 = cv2.imread(pathh+lst.fname[i1], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(pathh+lst.fname[i2], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(pathh+lst.fname[i3], cv2.IMREAD_GRAYSCALE)

        reconsImgforCV1 = DCTprocess(img1,Rlb,Rub,Clb,Cub,Clb2,Cub2)
        reconsImgforCV2 = DCTprocess(img2,Rlb,Rub,Clb,Cub,Clb2,Cub2)
        reconsImgforCV3 = DCTprocess(img3,Rlb,Rub,Clb,Cub,Clb2,Cub2)
        # np.savetxt('./re/outputMat.txt', reconsImgforCV1, fmt='%.2e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

        img11 = cv2.resize(img1, (226, 303))
        img22 = cv2.resize(img2, (226, 303))
        img33 = cv2.resize(img3, (226, 303))
        re11 = cv2.resize(reconsImgforCV1, (226, 303))
        re22 = cv2.resize(reconsImgforCV2, (226, 303))
        re33 = cv2.resize(reconsImgforCV3, (226, 303))
        cv2.imshow('original_image1', img11)
        cv2.imshow('original_image2', img22)
        cv2.imshow('original_image3', img33)
        cv2.imshow('re1',re11)
        cv2.imshow('re2',re22)
        cv2.imshow('re3',re33)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    return 0


if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    main(patH, patH2)
