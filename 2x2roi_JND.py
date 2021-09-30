import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
# from matplotlib.pyplot import figure

# This is description
    # calculate contrast
    # based on the center of mass of extracted defects

basePath = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/2nd/0826_2nd/"
csvPath = basePath + "table4.csv" # table4 is sorted through JND

def bdcoord(img):
    # This function finds the boundary of the LCD
    _, img3 = cv2.threshold(img,127,255,0)
    x_min,x_max,y_min,y_max = 0,len(img[0]),0,len(img[:,0])
    
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

for ii in range(104,106,1): # img #1 to img #106 are JND below 3.5 (3.5 not included)
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
        # img1 is .tif (should be 10 bit, but the image is stored in 16 bit)
        # img2 is .bmp
        # we use img2 to find the approximate location of defect
    
    print("===============")
    print(lst.Chip_ID[ii])
    print("Img #" + str(ii))

    # obtain the boundary of the LCD and crop it
    x_min, x_max, y_min, y_max = bdcoord(img1)
    img4 = img2[y_min//2:y_max//2,x_min//2:x_max//2]
    img1 = img1[y_min:y_max,x_min:x_max]

    # find the Blue pixels (the bounding box) from the predicted img
    indices = np.where(np.all(img4 == [0,0,255], axis=-1))
    coords = zip(indices[0], indices[1])
    a = list(coords)[0]
    
    # DCT
    imgFloat = img1.astype('float')
    coeff = cv2.dct(imgFloat)
    coeff[:,3:]=0
    coeff[3:][:]=0
    reconsImg = cv2.idct(coeff)
    diff = reconsImg-img1
    # print(img1[152:158:,152:158])
    # print(diff[152:158:,152:158].min())
    # print(reconsImg[152:158:,152:158])
    # diff could have negative values in it

    y_bd = min(2*a[0]+224,y_max)
    x_bd = min(2*a[1]+224,x_max)
    img1 = img1[2*a[0]:y_bd,2*a[1]:x_bd]
    diff = diff[2*a[0]:y_bd,2*a[1]:x_bd]
    diff_n = diff + (-1)**(diff.min()>0)*abs(diff.min())
    # opencv adaptive threshold takes 8 bit as input
    thresh1 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 22)# obtain white Mura
    thresh2 = cv2.adaptiveThreshold(diff_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45,-22)# obtain black Mura
    BothT = thresh1+thresh2
    BothT2 = cv2.medianBlur(BothT, 9)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(BothT2, cv2.MORPH_OPEN, kernel)
    
    plt.imshow(opening,cmap='gray')
    plt.show()

    color = (255,200,200)#(200,200,200)
    lst_dfcts = []
    lst_bdbox = []

    _, opening2 = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    flag = 0
    if len(contours) == 0: # This if-else should be replaced by better CV algorithms in later versions
        continue
    else:
        for i in reversed(range(len(contours))):
            cimg,cimg2 = np.zeros_like(opening), np.zeros_like(opening)
            cv2.drawContours(cimg, contours, i, color, -1, cv2.LINE_8)
            pts = np.where(cimg == 255)
            lst_dfcts.append(img1[pts[0], pts[1]]) # record the intensity at the defect pixels
            bdRect = cv2.boundingRect(contours[i])
            cv2.rectangle(cimg2, (int(bdRect[0]), int(bdRect[1])),\
                        (int(bdRect[0]+bdRect[2]), int(bdRect[1]+bdRect[3])), 255, -1)
            pts2 = np.where(cimg2 == 255)
            lst_bdbox.append(img1[pts2[0], pts2[1]])

            M = cv2.moments(contours[i])
            
            if M["m00"]!=0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # if flag == 0 and abs(cY-57)<10:
                #     print(cX,cY)
                if abs(cY-57)<15 and abs(cX-55)<15: # "and" is NOT same as &
                    print(cX,cY)
                    break
            else:
                cX,cY = 0,0
    
    total_pixel_count = img1.shape[0]*img1.shape[1]
    defect_pixel_count = 0
    for i in range(len(lst_dfcts)):
        defect_pixel_count += len(lst_dfcts[i])

    defect_intens = 0
    for i in range(len(lst_dfcts)):
        defect_intens += sum(lst_dfcts[i])
    total_intens = cv2.sumElems(img1)[0] - defect_intens

    I_Back = total_intens/(total_pixel_count-defect_pixel_count)
    roisum = cv2.sumElems(img1[cY:cY+1,cX:cX+1])
    I_new = roisum[0]*255/(4*abs(roisum[0]/4-I_Back))
    IB_new = I_Back*255/(abs(roisum[0]/4-I_Back))
    Contrast = abs(I_new-IB_new)/(I_new+IB_new)

    # I_Back = total_intens/(total_pixel_count-defect_pixel_count)
    # Contrast = abs(img1[cY,cX]-I_Back)/(img1[cY,cX]+I_Back)
    with open("re/reN1.csv", 'a') as f:
        if ii == 0:
            f.write("Pred, Real\n")
        f.write(str(Contrast)+", "+str(lst.RealJND[ii])+"\n")