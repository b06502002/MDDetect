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
    for ii in range(256):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        imgFloat = img1.astype('float')
        coeff = cv2.dct(imgFloat)
        npary += abs(coeff)/256

    # np.savetxt("./re/fanalysis.csv", npary, delimiter=",")
    
    return np.transpose((abs(npary)>1000).nonzero())

if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    print(main(patH, patH2))
