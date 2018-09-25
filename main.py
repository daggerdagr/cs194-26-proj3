import numpy as np
import skimage as sk
import skimage.io as skio
import os
import datetime

# custom packages
from utils import *
from operations import apply, Operation


### INPUT

# name of the input file1
# impath1 = 'sample_imgs/girlface.bmp'
impath1 = 'sample_imgs/DerekPicture.jpg'
imname1 = "".join(os.path.basename(impath1).split(".")[:-1])

# name of the input file2
impath2 = 'sample_imgs/girlface.bmp'
imname2 = "".join(os.path.basename(impath2).split(".")[:-1])


# output fileName
fOutputDirectory = "output_imgs"
currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
fname = "%s_%s_%s" % (currTime, imname1, "result")
fFormat = ".bmp"

# read in the image
im1 = skio.imread(impath1)
im2 = skio.imread(impath2)

# convert to double (might want to do this later on to save memory)
im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

if im1.ndim == 2:
    im1 = grayscale2RGB(im1)
if im2.ndim == 2:
    im2 = grayscale2RGB(im2)

### LOGIC

# result = apply(Operation.Unsharp, im, None)
result = apply(Operation.GaussBlur_3D, im1, None)

### RESULT

im_out = result

### SAVE OUTPUT

# save and display the image
printImage(fOutputDirectory + "/" + fname + fFormat, im_out, disp=True)