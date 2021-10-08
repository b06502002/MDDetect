import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CropImg(IMG):
    canny_output = cv2.Canny(IMG, 10, 255)
    
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    print(len(contours))
    for i in range(len(contours)):
        color = (100,100,255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    
    cv2.imshow('Contours', drawing)
    return 0

def main(pathh,pathtoimg):
    while(1):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        img1 = cv2.imread(pathh+lst.fname[310], cv2.IMREAD_GRAYSCALE)

        CropImg(img1)

        cv2.imshow('1',img1)
        key = cv2.waitKey(1)
        if key == 27:
            break
    return 0


if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_data/"
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    main(patH, patH2)
