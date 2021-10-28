import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(patH):
    blank_image = np.zeros((512,512,1), np.uint8)
    for i in range(16):
        if i%2 == 0:
            blank_image[128+16*i:128+16*(i+1),128:384] = 255

    cv2.imwrite(patH+'stripe_but_largeImg_centered.png', blank_image)

    return 0

if __name__ == "__main__":
    # patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/"
    # patH2 = "/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/"
    patH = "/home/cov/Desktop/codefiles/python files here/projects/PML/PML mura (sync)/opcv_dct_test/"
    main(patH)