import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from image_proc import *
from detect_corners import *

def scan_dim(im,dim):
    return np.sum(im == 0,axis=dim)

def scan_trim(arr):
    return np.trim_zeros(arr)

def extract(im, y,m,dim):
    if dim == 0:
        im = np.transpose(im)
    
    newim = np.zeros(im.shape,dtype=np.uint8)
    lastscan = None
    for i,val in enumerate(y):
        if val == m:
            lastscan = im[i]
            break
    for i in range(0,len(y)):
        if y[i] == m:
            newim[i] += im[i]
            lastscan = np.copy(im[i])
            im[i] = 255
        else:
            newim[i] += lastscan

    if dim == 0:
        im = np.transpose(im)
        newim = np.transpose(newim)

    return newim,im


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    dim = 0

    y = scan_dim(arr,dim)
    m = stats.mode(y)
    print('%d occurs %d times (%.2f)' % (m[0][0],m[1][0], float(m[1][0])/len(y) ))
    print(m)


    if (len(y)/m[0][0]) > 10:
        print('It\'s a line!  Extracting..')
        newim,arr = extract(arr,y,m[0][0],dim)
    save(arr,'output1.png')
    save(newim,'output2.png')
    #plt.plot((y==m[0][0]) * 50 + m[0][0])
    #plt.plot(y)
    #plt.show()




