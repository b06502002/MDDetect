import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

# This is description
    # sort same images with same JND in the same file

basePath = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/2nd/0826_2nd/"
csvPath = basePath + "table4.csv"


def bdcoord(img):
    # This function finds the boundary of the LCD
    _, img3 = cv2.threshold(img,127,255,0)
    x_min,x_max,y_min,y_max = 0,len(img[0]),0,len(img[:,0])
    ct = 0
    
    for i in range(len(img3[:,0])):
        if np.count_nonzero(img3[i,:]==255) > len(img3[:,0])*2//3:
            if i%2==1:
                y_min = i+1
            else:
                y_min = i
            break
    for i in reversed(range(len(img3[:,0]))):
        if np.count_nonzero(img3[i,:]==255) > len(img3[:,0])*2//3:
            if i%2==1:
                y_max = i-1
            else:
                y_max = i
            break

    for i in range(len(img3[0])):
        if np.count_nonzero(img3[:,i]==255) > len(img3[0])//3:
            if i%2==1:
                x_min = i+1
            else:
                x_min = i
            break
    for i in reversed(range(len(img3[0]))):
        if np.count_nonzero(img3[:,i]==255) > len(img3[0])//3:
            if i%2==1:
                x_max = i-1
            else:
                x_max = i
            break
    
    return x_min, x_max, y_min, y_max

lst = pd.read_csv(csvPath)

for ii in range(0,126,1):
    imgPath = basePath + lst.Deftype[ii] +'/'+ lst.Chip_ID[ii] +'/'
    if lst.Deftype[ii] == 'WSL128':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)

    elif lst.Deftype[ii] == 'WSL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)

    elif lst.Deftype[ii] == 'DWL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)

    elif lst.Deftype[ii] == 'BSL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)

    elif lst.Deftype[ii] == 'BSL128':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)

    elif lst.Deftype[ii] == 'IRR':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
    # obtain variables img1 and img2
        # img1 is .tif
        # img2 is .bmp
        # we use img2 to find the approximate location of defect
    
    print("===============")
    print(lst.Chip_ID[ii])
    print(lst.RealJND[ii])
    print("Img #" + str(ii))

    with open("re4_max_gray.csv", 'a') as f:
            f.write(str(np.amax(img1))+"\n")
    
