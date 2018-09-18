import numpy as np
import skimage as sk
import skimage.io as skio
import os

# custom packages
from utils import *
from operations import apply, Operation


### INPUT

# name of the input file
print(os.listdir("sample_imgs"))
impath = 'sample_imgs/girlface.bmp'
imname = os.path.basename(impath)

# output fileName
fOutputDirectory = "output_imgs"
fname = imname + "result"
fFormat = ".jpg"

# read in the image
im = skio.imread(impath)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

### LOGIC

apply(Operation.Unsharp, im, None)





### RESULT

im_out = im ## OUTPUT

### SAVE OUTPUT

# save and display the image
printImage(fOutputDirectory + "/" + fname + fFormat, im_out, disp=True)