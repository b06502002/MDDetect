import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

for ii in range(90,96,1):
    imgPath = basePath + lst.Deftype[ii] +'/'+ lst.Chip_ID[ii] +'/'
    if lst.Deftype[ii] == 'WSL128':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)

    elif lst.Deftype[ii] == 'WSL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)

    elif lst.Deftype[ii] == 'DWL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)

    elif lst.Deftype[ii] == 'BSL48':
        for file_name in os.listdir(imgPath):
            if ("L48" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)

    elif lst.Deftype[ii] == 'BSL128':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)

    elif lst.Deftype[ii] == 'IRR':
        for file_name in os.listdir(imgPath):
            if ("L128" in file_name) and (".tif" in file_name):
                fname = imgPath + file_name
                img1 = tiff.imread(fname)
            if ".bmp" in file_name:
                fname = imgPath + file_name
                img2 = cv2.imread(fname)
    # obtain variables img1 and img2
        # img1 is .tif
        # img2 is .bmp
        # we use img2 to find the approximate location of defect
    
    print("===============")
    print(lst.Chip_ID[ii])
    print(lst.RealJND[ii])
    print("Img #" + str(ii))
    x_min, x_max, y_min, y_max = bdcoord(img1)
    img4 = img2[y_min//2:y_max//2,x_min//2:x_max//2]

    indices = np.where(np.all(img4 == [0,0,255], axis=-1)) # find the Blue pixels (the bounding box)
    coords = zip(indices[0], indices[1])
    a = list(coords)[0]

    img1 = img1[y_min:y_max,x_min:x_max]
    
    y_bd = min(2*a[0]+224,y_max)
    x_bd = min(2*a[1]+224,x_max)
    img1 = img1[2*a[0]:y_bd,2*a[1]:x_bd]
    # plt.imshow(img1)
    # plt.show()

    directory = str(lst.RealJND[ii]).replace(".",'_')
    parent_dir = "../img_not_for_sync/"
    pathtoimg = os.path.join(parent_dir, directory)

    if not os.path.exists(pathtoimg):
        os.mkdir(pathtoimg)
    print(img1[52:58,52:58])

    # img8bit = (img1).astype('uint8')


    # img1_float = img1.astype('float')
    # img1_float -= img1_float.min()
    # img1_float /= img1_float.max()
    # img1_float *= 255
    # img8bit = img1_float.astype('uint8')

    img1_float = img1.astype('float')
    img1_float -= img1_float.min()
    img1_float /= img1.max()
    img1_float *= 255
    img8bit = img1_float.astype('uint8')

    print(img8bit[52:58:,52:58])
    #cv2.imwrite(pathtoimg+"/"+str(lst.Chip_ID[ii])+".png", img8bit)

    # with open("re1.csv", 'a') as f:
    #     if ii == 0:
    #         f.write("Pred, Real\n")
    #     f.write(str(Contrast)+", "+str(lst.RealJND[ii])+"\n")