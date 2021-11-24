import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nothing(x):
    pass

def GetContour(IMG):
    IMGGB = cv2.GaussianBlur(IMG, (3,3), 0)
    canny_output = cv2.Canny(IMGGB, 0, 255)

    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing = IMG
    lst_dfcts = []

    for i in range(len(contours)):
        # color = (100,100,255)
        color = 255
        cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_8, hierarchy, 0, (-3,3)) # specifically for cell lines at certain position

        # deprecated
            # pts = np.where(drawing == 255)     # record the boundary pixels (Contours)
            # lst_dfcts.append((pts[0], pts[1])) # stores the boundary pixels
        
    # obtain pixels in contours
    if len(contours)==0:
        return drawing, []
    else:
        lst_coord = str(contours[0]).replace("[[[",'').replace("]]]",'').replace("[[",',').replace("]]",',').replace(" ",',').replace("\n\n",'').replace(",,",',').split(',')
        lst_pts = []
        for i in range(len(contours[0])):
            lst_pts.append((contours[0][i][0][0],contours[0][i][0][1]))
        return drawing, lst_pts

def test(pathh):
    csvPath = pathh + "filenames.csv"
    lst = pd.read_csv(csvPath)
    ct_for_acc = 0
    ct1 = 0
    for ii in range(len(lst)): # iterate through different images
        img1 = cv2.imread(pathh+lst.fname[ii], cv2.IMREAD_GRAYSCALE)
        ctur, pt = GetContour(img1)
        sum1, comparison = (0,0)
        for jj in range(len(pt)):
            sum1 += img1[pt[jj][1],pt[jj][0]]
            if pt[jj][1]-3 < 0 or pt[jj][0]+3 > 903:
                comparison += 0
                ct1 += 1
            else:
                comparison += img1[pt[jj][1]-3,pt[jj][0]+3]

        if len(pt)-ct1==0 or len(pt)==0:
            print(sum1>comparison)
            # print(ii,"is cell line:", sum1>comparison)
        else:   
            print((sum1/len(pt))>comparison/(len(pt)-ct1))
            # print(ii,"is cell line:",(sum1/len(pt))>comparison/(len(pt)-ct1))         
        if sum1>comparison:
            ct_for_acc += 1
        # cv2.imshow('image',img1)
        # cv2.waitKey(0)
        # print(img1)
    
    print("prediction accuarcy: ",ct_for_acc/len(lst),"# cell line: ",ct_for_acc)
    return 0

def main(pathh,pathtoimg):

    cv2.namedWindow('trackbar')
    cv2.createTrackbar('Img1', 'trackbar', 0, 53, nothing)# Total 310 # Cell line 53

    while(1):
        csvPath = pathh + "filenames.csv"
        lst = pd.read_csv(csvPath)
        i1 = cv2.getTrackbarPos('Img1', 'trackbar')
        img1 = cv2.imread(pathh+lst.fname[i1], cv2.IMREAD_GRAYSCALE)
        img11 = cv2.resize(img1, (226, 303))

        dr = GetContour(img1)
        dr2 = cv2.resize(dr, (452, 606))
        cv2.imshow('re', dr2)
        cv2.imshow('original image', img11)

        key = cv2.waitKey(1)
        if key == 27:
            break
    return 0


if __name__ == "__main__":
    patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/" #mura_image_CL/"#
    patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/cl/"
    #main(patH, patH2)
    test(patH)
