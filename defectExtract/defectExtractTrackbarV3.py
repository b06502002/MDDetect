import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nothing(x):
    pass

def DFTprocess(IMG,lll0):

    f = np.fft.fft2(IMG)
    fshift = np.fft.fftshift(f)
    
    for jj in range(len(lll0)):
        fshift[int(lll0.C1[jj])][int(lll0.C2[jj])] = 0
    img_back = cv2.idft(fshift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back, 0



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

def main(pathh,path2):

    cv2.namedWindow('trackbar')
    cv2.createTrackbar('Img1', 'trackbar', 0, 256, nothing)# Total 310 # Cell line 53
    cv2.createTrackbar('coeff_threshold', 'trackbar', 0, 1000, nothing)
    cv2.createTrackbar('adaptive_c','trackbar', 0,100, nothing)
    cv2.createTrackbar('adaptive_kernelsize', 'trackbar', 0,100, nothing)
    cv2.createTrackbar('median_kernelsize', 'trackbar', 0,30, nothing)
    cv2.createTrackbar('opening_kernelsize', 'trackbar', 0,30, nothing)
    cv2.createTrackbar('median_kernelsize2', 'trackbar', 0,30, nothing)

    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)

    IndexLst = pd.read_csv(path2)

    while(1):
        coeffThres = cv2.getTrackbarPos('coeff_threshold','trackbar')
        adpc = cv2.getTrackbarPos('adaptive_c','trackbar')
        adpk = cv2.getTrackbarPos('adaptive_kernelsize','trackbar')
        medk = cv2.getTrackbarPos('median_kernelsize','trackbar')
        medk2 = cv2.getTrackbarPos('median_kernelsize2','trackbar')
        opk = cv2.getTrackbarPos('opening_kernelsize', 'trackbar')

        i1 = cv2.getTrackbarPos('Img1', 'trackbar')
        img1 = cv2.imread(pathh+lst.fname[i1], cv2.IMREAD_GRAYSCALE)

        reconsImgforCV1, _ = DFTprocess(img1, IndexLst)
        thresImg = thres(reconsImgforCV1, adpc, 2*adpk+3, 2*medk+1, 2*medk2+1, opk+2)
        # np.savetxt('./re/outputMat.txt', reconsImgforCV1, fmt='%.2e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

        img11 = cv2.resize(img1, (226, 303))
        re11 = cv2.resize(reconsImgforCV1, (452, 606))
        re12 = cv2.resize(thresImg, (452, 606))
        cv2.imshow('original_image1', img11)
        cv2.imshow('re1',re11)
        cv2.imshow('re2', re12)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/re/fanalysisIndices.csv"
    main(patH, patH2)
