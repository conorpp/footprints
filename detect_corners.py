import sys,os,json,argparse
import matplotlib as plt
from PIL import Image, ImageDraw

import numpy as np


if len(sys.argv) != 3:
    print 'usage: %s <input.png> <output>' % sys.argv[0]
    sys.exit(1)

im = Image.open(sys.argv[1])
output = sys.argv[2]


arr = np.array(im)
if arr.shape[2] == 4:
    arr = np.delete(arr, 3, 2) # get rid of alpha..

colsum = arr.shape[0] * 255 * 3

def trim(im):
    # trim columns left
    while sum(im[:,0].flatten()) == colsum:
        im = np.delete(im, 0, 1)

    # trim columns right
    while sum(im[:,-1].flatten()) == colsum:
        im = np.delete(im, -1, 1)

    rowsum = im.shape[1] * 255 * 3

    # trim rows top
    while sum(im[0].flatten()) == rowsum:
        im = np.delete(im, 0, 0)

    # trim rows bottom
    while sum(im[-1].flatten()) == rowsum:
        im = np.delete(im, -1, 0)

    return im

out = trim(arr)

out = Image.fromarray(out)
out.save(output)

