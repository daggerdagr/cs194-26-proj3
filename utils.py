from skimage.io import imsave, imshow, show
import numpy as np

#IMAGE IO
TESTING_DIR = "trash_imgs"

def printImage(path, im, disp=True):
    # save the image
    imsave(path, im)

    # display the image
    if disp:
        viewImage(im)

def testImage(path, im, disp=True):
    path = TESTING_DIR + "/" + path

    printImage(path, im, disp)

def viewImage(im):
    imshow(im)
    show()

def grayscale2RGB(im):
    assert im.ndim == 2

    return np.dstack((np.dstack((im, im)), im))
