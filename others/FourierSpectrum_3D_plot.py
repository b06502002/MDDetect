import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


### 
    # This py script tests if the matplotlib (or other packages) can plot the desired spectral domain in 3D view
    # The code tested here can be copied and used by other programs, or just be as a reference


def readIMGmode(path, ch = 0):
    if ch == 0:
        return cv.imread(path, cv.IMREAD_GRAYSCALE)
    elif ch == 'b':
        return cv.imread(path, cv.IMREAD_UNCHANGED)[:,:,0]
    elif ch == 'g':
        return cv.imread(path, cv.IMREAD_UNCHANGED)[:,:,1]
    elif ch == 'r':
        return cv.imread(path, cv.IMREAD_UNCHANGED)[:,:,2]
    elif ch == 'color':
        return cv.imread(path, cv.IMREAD_UNCHANGED)
    else:
        print("Invalid mode\n")
        return False

def FFTshow(IMG):
    f = np.fft.fft2(IMG)
    fshift = np.fft.fftshift(f)
    # phase_spectrumR = np.angle(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.figure()

    ax = plt.axes(projection='3d')
    y = range( IMG.shape[0] )
    x = range( IMG.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface( X, Y, magnitude_spectrum, cmap=plt.cm.coolwarm )

    plt.show()
    return 0

def FFTshow2(IMG, IMG2):
    f = np.fft.fft2(IMG)
    fshift = np.fft.fftshift(f)
    # phase_spectrumR = np.angle(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    f2 = np.fft.fft2(IMG2)
    fshift2 = np.fft.fftshift(f2)
    magnitude_spectrum2 = 20*np.log(np.abs(fshift2))
    plt.figure()

    ax = plt.axes(projection='3d')
    y = range( IMG.shape[0] )
    x = range( IMG.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface( X, Y, magnitude_spectrum-magnitude_spectrum2, cmap=plt.cm.coolwarm )

    plt.show()
    return 0

def enhance(IMG,brightness=365,contrast=217):
    """
    this funciton increases the brightness and the contrast
    ref: https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/
    """
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness > 0:
        shadow = brightness
        max = 255
    else:
        shadow = 0
        max = 255 + brightness
    al_pha = (max - shadow) / 255
    ga_mma = shadow
    cal = cv.addWeighted(IMG, al_pha, IMG, 0, ga_mma)

    Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    Gamma = 127 * (1 - Alpha)
    return cv.addWeighted(cal, Alpha, cal, 0, Gamma)

def show(IMG,name='Do not click X to close this window'):
    """
    Show 2D images
    """
    cv.imshow(name,IMG)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0

def main(path,mode = 0):
    img = readIMGmode(path,mode)
    print(img.shape)
    # show(img)
    show(enhance(img))
    # FFTshow(img)
    # FFTshow(enhance(img))
    # FFTshow2(enhance(img),img) # enhance(img) make the whole panel white
    return 0


if __name__ == "__main__":
    PatH = "/home/cov/Desktop/PML/project1_Mura/AUO_Data/maskrcnn_label_data/mura_image_BandBlob/2L_128_sImage_Image2_063.jpg"
    main(PatH)