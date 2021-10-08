import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nothing(x):
    pass

def GetContour(IMG):
    IMGGB = cv2.GaussianBlur(IMG, (3,3), 0)
    canny_output = cv2.Canny(IMGGB, 0, 255)

    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    lst_dfcts = []

    for i in range(len(contours)):
        color = (100,100,255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        pts = np.where(drawing == 255)
        lst_dfcts.append(IMG[pts[0], pts[1]])
    return drawing

def thres(IMG, adp_c, adp_k, med_k, med_k2, op_k):
    diff_n = cv2.normalize(IMG, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    thresh1 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adp_k, adp_c)# obtain white Mura
    thresh2 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adp_k,-adp_c)# obtain black Mura
    BothT = thresh1+thresh2
    BothT2 = cv2.medianBlur(BothT, med_k)
    kernel = np.ones((op_k,op_k),np.uint8)
    opening = cv2.morphologyEx(BothT2, cv2.MORPH_OPEN, kernel)
    opening2 = cv2.medianBlur(opening, med_k2) 
    return opening2

def main(pathh,pathtoimg):

    cv2.namedWindow('trackbar')
    cv2.createTrackbar('Img1', 'trackbar', 0, 53, nothing)# Total 310 # Cell line 53

    while(1):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        i1 = cv2.getTrackbarPos('Img1', 'trackbar')
        img1 = cv2.imread(pathh+lst.fname[i1], cv2.IMREAD_GRAYSCALE)
        img11 = cv2.resize(img1, (226, 303))

        dr = GetContour(img1)
        dr2 = cv2.resize(dr, (452, 606))
        cv2.imshow('re', dr2)
        cv2.imshow('original image', img11)

        key = cv2.waitKey(1)
        if key == 27:
            break
    return 0


if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_CL/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    main(patH, patH2)
