import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nothing(x):
    pass

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def main(pathh,pathtoimg):
    npary = np.zeros((1212,904))
    for ii in range(234,241,1):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        plt.imshow(img1,cmap='gray')
        plt.show()
        imgFloat = img1.astype('float')
        coeff = cv2.dct(imgFloat/255)
        npary += abs(coeff)*255/7

    # plt.imshow(cv2.idct(npary),cmap='gray')
    # plt.show()

    np.savetxt("./re/fanalysis1109_4.csv", npary, delimiter=",")
    
    # with open("./re/fanalysisIndices.csv", "w") as f:
    #     f.write("C1,C2\n")
    #     np.savetxt(f, np.transpose((abs(npary)>20).nonzero()), delimiter=",")

    return 0

def singleImgTr(pathh): # obtain frequency map from a single image (saves as a csv file)
    img1 = cv2.imread(pathh, cv2.IMREAD_GRAYSCALE)
    imgFloat = img1.astype('float')
    coeff = cv2.dct(imgFloat)
    np.savetxt("./re/fanalysis1110_2.csv", coeff, delimiter=",")

    return 0

def singleImgTr2(pathh,pathh2): # obtain frequency map from a residual image (saves as a csv file)
    img1 = cv2.imread(pathh, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(pathh2, cv2.IMREAD_GRAYSCALE)
    imgFloat1 = img1.astype('float')
    imgFloat2 = img2.astype('float')
    imgFloat = imgFloat1 - imgFloat2
    imgFloat -= imgFloat.min()
    imgFloat /= imgFloat.max()
    coeff = cv2.dct(imgFloat)
    # plt.imshow(cv2.idct(coeff),cmap='gray')
    # plt.show()
    # np.savetxt("./re/fanalysis1110_3.csv", coeff, delimiter=",")
    reimg = cv2.idct(coeff)
    reimg -= reimg.min()
    reimg /= reimg.max()
    reimg *= 255
    reimg = reimg.astype('uint8')
    cv2.imwrite("./re/blob_re2.png",reimg)
    return 0


if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    # main(patH, patH2)
    patHH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/L48_sImage_Image3_000 (87).jpg"
    patHH2 = "/home/cov/Desktop/PML/project1_Mura/gimp img/L48_sImage_Image3_000 (89)_PS.png"
    singleImgTr2(patHH,patHH2)