import skimage as sk
import skimage.filters as sf
import numpy as np

from enum import Enum

## custom
from utils import *

class Operation(Enum):
    Unsharp = "unsharp"

def apply(operation, im1, im2):
    if operation == Operation.Unsharp:
        assert im2 == None
        return unsharpOp(im1)
    else:
        raise Exception("unrecognized operation : %s" % operation)

##################
# OPERATIONS
##################

def unsharpOp(im):
    gaussIm = sf.gaussian(im, 10)
    testImage("testGauss.jpg", gaussIm)
    result = im + (im - gaussIm)
    result = np.clip(result, -1, 1)

    testImage("testUnsharp.jpg", result)