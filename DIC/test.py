import cv2
import numpy as np
from numpy.core.fromnumeric import transpose
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def set_up_OF_field(G_t1, G_t2, window_size, tau=2e-2,stride=4):
    # kernels are for calculating (partial G/ partial x) and (partial G/ partial y)
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window

    # normalize pixels
    G_t1 = G_t1 / 255.
    G_t2 = G_t2 / 255.

    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    pG_px = signal.convolve2d(G_t1, kernel_x, boundary='symm', mode=mode) # using convolution to calculate gradient map
    pG_py = signal.convolve2d(G_t2, kernel_y, boundary='symm', mode=mode)
    Delt_G = signal.convolve2d(G_t2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(G_t1, -kernel_t, boundary='symm', mode=mode)
    u,v = np.zeros(G_t1.shape), np.zeros(G_t1.shape)
    
    # within window window_size * window_size
    for i in range(w, G_t1.shape[0]-w):
        if (i-w)%stride!=3:
            continue
        else:
            for j in range(w, G_t1.shape[1]-w):
                if (j-w)%stride!=3:
                    continue
                else:
                    pgpx = pG_px[i-w:i+w+1, j-w:j+w+1].flatten()
                    pgpy = pG_py[i-w:i+w+1, j-w:j+w+1].flatten()
                    deltag = Delt_G[i-w:i+w+1, j-w:j+w+1].flatten()

                    G = np.column_stack((pgpx, pgpy))
                    GTG = np.matmul(transpose(G),G)
                    GTg = np.matmul(transpose(G),deltag)

                    eig = np.linalg.eigvals(GTG)
                    if abs(eig.min())<tau:
                        deltaX = -tau*GTg
                    else:
                        deltaX = -np.matmul(np.linalg.inv(GTG),GTg)

                    u[i,j]=deltaX[0]
                    v[i,j]=deltaX[1]

    # set color for quiver plot
    # colors = np.arctan2(u, v)
    # norm = Normalize()
    # norm.autoscale(colors)
    # colormap = cm.inferno


    fig, ax = plt.subplots(figsize = (G_t1.shape[1]//4,G_t1.shape[0]//4))
    x,y = np.meshgrid(np.arange(G_t1.shape[1]//4),np.arange(G_t1.shape[0]//4))# the image is upside down
    ax.quiver(x,y,u[::4,::4],np.zeros(G_t1.shape)[::4,::4], np.arctan2(u[::4,::4], np.zeros(G_t1.shape)[::4,::4]), scale=1000, pivot='mid', color='g') #Numpy indexing follows a start:stop:stride convention
    # takes only u component (x direction movement are plotted)
    plt.gca().invert_yaxis()

    plt.figure()
    plt.imshow(u)


    # plt.figure()
    # plt.imshow(v)
    plt.show()
    return (u,v)


if __name__ == "__main__":
    g1 = cv2.imread("/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/tmp2/4L_48_sImage_Image4_068.jpg", cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imread("/home/cov/Desktop/codefiles/python files here/projects/PML/img_NFS_Contour/tmp2/4L_48_sImage_Image4_069.jpg", cv2.IMREAD_GRAYSCALE)

    set_up_OF_field(g1,g2,50,stride=4)