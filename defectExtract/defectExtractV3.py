import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(pathh,pathtoimg):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    for ii in range(1): #311
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        imgFloat = img1.astype('float')
        coeff = cv2.dct(imgFloat)
        coeff[:,:100]=0
        coeff[100:][:]=0
        reconsImg = cv2.idct(coeff)
        # diff = reconsImg-img1
        plt.imshow(reconsImg,cmap='gray')
        plt.show()
        

        print("==={}===".format(ii))
        # plt.imsave(pathtoimg+str(ii)+".bmp", coeff, cmap="gray")
        # plt.imsave(pathtoimg+"re"+lst.fname[ii],reconsImg,cmap="gray")
    return 0

def main2(pathh,pathtoimg):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    for ii in range(311):
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("==={}===".format(ii))
        for i in reversed(range(len(contours))):
            cimg = np.zeros_like(img1)
            cv2.drawContours(cimg, contours, i, 255, -1, cv2.LINE_8)
                        
        # plt.imsave(pathtoimg+str(ii)+".bmp", coeff, cmap="gray")
        plt.imsave(pathtoimg+"re"+lst.fname[ii], cimg,cmap="gray")

    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/" # "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Spectrum/"# 
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    main(patH, patH2)
    # main2(patH, patH2)
    