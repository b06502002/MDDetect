import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(pathh,pathtoimg):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    for ii in range(0,10,1):
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        imgFloat = img1.astype('float')
        coeff = cv2.dct(imgFloat)
        
        
        # coeff[:,:1]=0
        # coeff[:1][:]=0
        # coeff[0][0]=0
        # reconsImg = cv2.idct(coeff)
        #diff = reconsImg-img1
        # plt.imshow(coeff,cmap='gray')
        # plt.show()
        
        # img2 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        # imgFloat2 = img2.astype('float')
        # coeff2 = cv2.dct(imgFloat2)
        # reconsImg = cv2.idct(abs(coeff2-coeff))


        print("=====")
        # plt.imsave(pathtoimg+str(ii)+".bmp", coeff, cmap="gray")
        #plt.imsave(pathtoimg+"re"+str(ii)+".bmp",reconsImg,cmap="gray")
    return 0

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Spectrum/"
    main(patH, patH2)