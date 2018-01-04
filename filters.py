import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from utils import *
from analyzers import *


def pass_rectangles(rects):
    filtered = []
    left = []
    for x in rects:
        if x['area-ratio'] > 0.001:
            if x['conf'] > .95:
                filtered.append(x)
            else:
                left.append(x)
        else:
            left.append(x)
    return filtered,left

def pass_lines(rects):
    lines = []
    leftover = []
    #rects = sorted(rects, key = lambda x : x['line-conf'])
    for x in rects:
        #if (x['line-conf'] > .3) and (x['aspect-ratio'] > 3):
            #lines.append(x)
        #elif (x['line-conf'] > .9) and (x['aspect-ratio'] > 1.5):
            #lines.append(x)
        score = x['sum']['score']
        ran   = x['sum']['range']

        if x['aspect-ratio'] > .9:
            if score > .7 and ran < 4:
                lines.append(x)
            else:
                leftover.append(x)
        else:
            leftover.append(x)
    return lines,leftover
    #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    #for x in rects:
        #tlc = x['contour'][2]
        #brc = x['contour'][0]
        #w = brc[0] - tlc[0]
        #h = brc[1] - tlc[1]
        #if w < 4 or h < 4:
            #lines.append(x)
    #return lines

def contains_line(spec):
    s = spec['sum']
    return len(s['sum'])/s['mode'][0] > 15

def pass_potential_lines(inp):
    lines = []
    nolines = []
    for x in inp:
        if contains_line(x):
            lines.append(x)
        else:
            nolines.append(x)
    return lines,nolines

def block_dots(fresh):
    good = []
    for x in fresh:
        nz = np.count_nonzero(x['img'])
        z = x['img'].shape[0] * x['img'].shape[1] - nz
        if z > 1:
            good.append(x)
    return good

def pass_triangles(inp):
    tris = []
    notris = []
    for x in inp:
        if x['triangle-area-ratio']>.5:
            tris.append(x)
        else:
            notris.append(x)

    return tris,notris

def pass_ocr(inp):
    good = []
    bad = []
    for x in inp:
        if x['ocr-conf'] >= 60:
            good.append(x)
        else:
            bad.append(x)
    return good,bad


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    dim = 1

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




