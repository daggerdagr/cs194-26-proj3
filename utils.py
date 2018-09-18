from skimage.io import imsave, imshow, show

#IMAGE IO
TESTING_DIR = "trash_imgs"

def printImage(path, im, disp=True):
    # save the image
    imsave(path, im)

    # display the image
    if disp:
        imshow(im)
        show()

def testImage(path, im, disp=True):
    path = TESTING_DIR + "/" + path

    printImage(path, im, disp)