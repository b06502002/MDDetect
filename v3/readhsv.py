import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main(patH):
    frame = cv.imread(patH, cv.COLOR_BGR2HSV)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # Note that converting to HSV decreases the image size
    frame_threshold = cv.inRange(frame_HSV, (0, 24, 50), (100, 70, 255)) # (H,S,V)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gen_image = frame_gray * (frame_threshold//255) + 0 * (1-(frame_threshold//255))

    return gen_image

# if __name__ == "__main__":
#     patH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/2L_128_sImage_Image2_064.jpg" 
#     gen_image = main(patH)

#     plt.imshow(gen_image, "gray")
#     plt.show()