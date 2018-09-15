import numpy as np
import skimage as sk
import skimage.io as skio
import os

# name of the input file
impath = 'sample_imgs/joseph-gruenthal-1057768-unsplash.jpg'
imname = os.path.basename(impath)

# output fileName
fOutputDirectory = "output_imgs"
fname = imname + "result"
fFormat = ".jpg"

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)







### SAVE OUTPUT

# create a color image
im_out = im ## OUTPUT

# save the image
skio.imsave(fOutputDirectory + fname + fFormat, im_out)

# display the image
skio.imshow(im_out)
skio.show()