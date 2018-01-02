import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from utils import *
from analyzers import *

def separate_lines(inp):
    out = []

    for x in inp:
        new,old = extract(x['img'],x['sum']['sum'], x['sum']['mode'][0], x['vertical'])
        new = wrap_image(new,x)
        old = wrap_image(old,x)
        out.append(new)
        out.append(old)
        # try other direction
        #v = (x['vertical'] + 1) & 1
        #swhole = scan_dim(x['orig'],v)
        #strim = scan_trim(swhole)
        #m = stats.mode(strim)
        #if contains_line({'sum':{'sum':swhole, 'mode':m}}):
            #x['vertical'] = v
            #x['sum']['sum'] = swhole
            #x['sum']['mode'] = m
            #x['sum']['range'] = np.ptp(strim)
            #fresh.append(x)
        #else:
    return out


def extract(im, y,m,dim):
    if dim == 0:
        im = np.transpose(im)
    im = np.copy(im)
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
        #elif y[i] > m:
        else:
            #im[i] = 255
            #newim[i] += lastscan
            newim[i] = 255
            pass

    if dim == 0:
        im = np.transpose(im)
        newim = np.transpose(newim)

    return newim,im


def trim_image(arr):
    arr['img'],x,y = trim(arr['img'])
    arr['offset'][0] += x
    arr['offset'][1] += y

def trim_images(imgs):
    for x in imgs:
        trim_image(x)

def trim(im):
    padding = 2
    colsum = im.shape[0] * 255 * 1
    xoff = 0
    yoff = 0

    # trim columns left
    jl= 0
    while 0 not in im[:,jl+padding].flatten():
        jl+= 1
        if (jl + padding) == im.shape[1]: 
            jl = 0
            break
        #im = np.delete(im, 0, 1)
    
    # trim columns right
    jr = im.shape[1]-1
    while 0 not in im[:,jr - padding-1].flatten():
        jr -= 1
        if (jr - padding) == -1: 
            jr = im.shape[1] - 1
            break
    
    im = im[:,jl:(jr+1)]
    rowsum = im.shape[1] * 255 * 1

    # trim rows top
    it = 0
    while 0 not in (im[padding + it].flatten()):
        it += 1
        if (it + padding) == im.shape[0]:
            it = 0
            break

    ib = im.shape[0] - 1
    # trim rows bottom
    while 0 not in (im[ib - padding -1].flatten()):
        ib -= 1
        if (ib - padding) == -1: 
            ib = im.shape[0] - 1
            break

    im = im[it:(ib+1),:]

    return im,jl,it

def explore(arr,i,j):
    trackmap = np.zeros(arr.shape[:2], dtype=np.uint8)
    return explore_r(arr,i,j,trackmap)

def explore_r(arr,i,j,trackmap):
    imax = arr.shape[0]-1
    jmax = arr.shape[1]-1


    nodes_to_visit = [(i,j)]

    def checkout(i,j):
        #print i,j
        if arr[i][j] == 0:
            if not trackmap[i,j]:
                trackmap[i,j] = 1             # add to list
                nodes_to_visit.append((i,j))

    while len(nodes_to_visit):
        node = nodes_to_visit.pop()
        trackmap[node[0],node[1]] = 1

        i = node[0]
        j = node[1]
        #arr[i,j] = 128              # debug it

        # right
        if (j+1) <= jmax:
            checkout(i,j+1)

            # bottom-right
            if (i+1) <= imax:
                checkout(i+1,j+1)

            # top-right
            if (i-1) >= 0:
                checkout(i-1,j+1)

        # left
        if (j-1) >= 0:
            checkout(i,j-1)

            # bottom-right
            if (i+1) <= imax:
                checkout(i+1,j-1)

            # top-right
            if (i-1) >= 0:
                checkout(i-1,j-1)

        # top
        if (i - 1) >= 0:
            checkout(i-1,j)

        # bottom
        if (i + 1) <= imax:
            checkout(i+1,j)

    return trackmap


def extract_components(arr):
    img = arr['img']
    track_map = np.zeros(img.shape[:2],dtype=np.uint8)
    submaps = []
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] == 0:
                if not track_map[i,j]:
                    track_map[i,j] = 1
                    submap = explore(img,i,j)
                    track_map += submap
                    def mapping2greyscale(mapping):
                        mapping = np.array((mapping == 0) * 255, dtype=np.uint8)
                        return mapping
                    submap = mapping2greyscale(submap)
                    submap = wrap_image(submap,arr)
                    submaps.append( submap )

    return submaps

def separate_rectangle(arr):
    squ = arr['contour'][:]
    grow_rect_by_one(squ)

    outside = np.copy(arr['img'])

    cv2.fillPoly(outside, [squ], 255)

    outside = wrap_image(outside,arr)
    return outside

def separate_rectangles(inp):
    outsides = []
    for x in inp:
        outsides.append(separate_rectangle(x))
    return outsides


