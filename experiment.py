import sys,os,json,argparse,time
from PIL import Image, ImageDraw
from random import randint

from scipy import signal
import numpy as np
import cv2

import analyzers
from utils import *
from filters import *
from analyzers import *
from processors import *

from preprocessing import *

def add_feature(im, feat):
    if type(im) == type({}):
        im = im['img']

    off = feat['offset']
    feat = feat['img']

    for i in range(0,feat.shape[0]):
        for j in range(0,feat.shape[1]):
            im[i+off[1],j+off[0]] = feat[i,j]

def die(submaps,comment):
    for x in submaps:
        debug = np.zeros(orig.shape,dtype=np.uint8) + 255
        add_feature(debug,x)
        save(debug,'out/%sitem%d.png' % (comment,x['id']))
        print('debugging',x['id'])
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    arr = remove_alpha(arr)
    arr = polarize(arr)
    arr = preprocess(arr)

    orig = np.copy(arr)

    #print(len(rectangles),'rectangles')
    #for i,r in enumerate(rectangles):
        #r = np.array(r)


    save(orig,'output.png')




