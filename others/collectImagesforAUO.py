import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    return reconsImgforCV, cv2.dct(imgFloat)


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


#      / Clb,Cub,Clb2,Cub2,Rlb,Rub, adpc,adpk,medk,opk,medk2
# 0-9: 3,400,0,0,0,3,5,24,0,5,0
# 10-26: 3,400,0,0,0,38,4,33,4,0,7
# 27-30: 36,415,0,0,0,26,4,21,4,3,2
# 31-109: 36,665,0,0,0,34,9,19,3,3,19
# 110-111: 0,0,0,84,0,27,7,19,3,16,6
# 112-138: 65,76,0,0,0,60,3,17,3,2,8
# 139-141: 65,76,0,0,0,60,2,19,4,2,27
# 142-160: 4,75,0,0,0,27,2,19,8,7,17
# 161-166: 6,75,0,0,0,60,2,15,20,20,20
# 167-184: 62,85,135,1000,0,30,4,34,8,7,10
# 185-194: 7,85,135,1000,0,60,7,10,7,7,14
# 195-211: 0,0,0,0,0,32,1,19,0,4,6
# 212-230: 135,1000,0,0,0,43,21,30,3,2,8
# 231-256: 65,76,135,1000,0,30,5,48,8,7,10

def main(pathh,pathtoimg):
    Clb,Cub,Clb2,Cub2,Rlb,Rub, adpc,adpk,medk,opk,medk2 = 65,76,135,1000,0,30,5,48,8,7,10
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    for ii in range(231,256,1):
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        reconsImgforCV1, _ = DCTprocess(img1,Rlb,Rub,Clb,Cub,Clb2,Cub2)
        thresImg = thres(reconsImgforCV1, adpc, 2*adpk+3, 2*medk+1, 2*medk2+1, opk+2)

        cv2.imwrite('./re/'+lst.fname[ii].split('.')[0]+'re.png',thresImg)
    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    main(patH, patH2)
