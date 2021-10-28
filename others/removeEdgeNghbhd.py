import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(pathh):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    
    for ii in range(200,207,1): #311
        print(ii)
        
    return 0

def main2(pathh,pathtoimg):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    for ii in range(10):
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        cv2.imshow('img',img1)
        cv2.waitKey(0)
        cv2.destroyWindow('img')
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("==={}===".format(ii))
        for i in reversed(range(len(contours))):
            cimg = np.zeros_like(img1)
            cv2.drawContours(cimg, contours, i, 126, -1, cv2.LINE_8)
            cv2.imshow('img',cimg)
            cv2.waitKey(0)
            cv2.destroyWindow('img')
        # plt.imsave(pathtoimg+str(ii)+".bmp", coeff, cmap="gray")
        # plt.imsave(pathtoimg+"re"+lst.fname[ii], cimg,cmap="gray")

    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/" # "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Spectrum/"# 
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/re/"
    main2(patH, patH2)
    