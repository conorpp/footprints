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
        #try:
            new,old = extract(x['img'],x['sum']['sum'], x['sum']['mode'][0], x['vertical'])
            new = wrap_image(new,x)
            old = wrap_image(old,x)
            out.append(new)
            out.append(old)
        #except Exception as e:
            #print('execption',e)
            #save_history(x)
            #sys.exit(1)

    return out

def get_mode_locations(y,val):
    locs = []
    start = None
    end = None
    for i,p in enumerate(y):
        if p == val:
            if start is None:
                start = i
                end = i+1
            else:
                end += 1
        else:
            if start is not None:
                locs.append([start,end])
                start = None
    return locs

# determines the starting position of mode 
# pixels and checks that they are consecutive
def mode_start_mode(im,locs,m):
    # scanning across 2nd dim to find m consecutive black pixels
    starts = []
    s = 0
    p = 0
    for loc in locs:
        for i in range(loc[0],loc[1]):
            p = 0
            s = 0
            for j in range(im.shape[1]):
                if im[i,j] == 0:
                    p += 1
                    if p == m:
                        starts.append(s)
                        break
                else:
                    if p:
                        break
                    s += 1

    #print('black pixels:', count_black(im))
    #print(starts,m)
    #print(locs)
    if len(starts):
        return stats.mode(starts)[0][0]
    else:
        return -1

def extend_locs(im,locs,m,start):
    for loc in locs:
        while True:
            white_both_sides = (im[loc[1],start-1] == 255) and (im[loc[1],start+m] == 255)
            mode = sum(im[loc[1],start:(start+m)]) == 0
            if white_both_sides and mode:
                loc[1] += 1
            else:
                break

def get_line_locations(im,y,line_start,m):
    locs = []
    start = None
    end = None
    for i,p in enumerate(y):
        if p >= m:

            white_both_sides = (im[i,line_start-1] == 255) and (im[i,line_start+m] == 255)
            mode = np.sum(im[i,line_start:(line_start+m)]) == 0

            if white_both_sides and mode:
                if start is None:
                    start = i
                    end = i+1
                else:
                    end += 1
            else:
                if start is not None: # redund
                    locs.append((start,end))
                    start = None
        else:
            if start is not None: #redund
                locs.append((start,end))
                start = None
    return locs


def extract(im, y,m,dim):
    if dim == 0:
        im = np.transpose(im)
    im = np.copy(im)
    newim = np.zeros(im.shape,dtype=np.uint8) + 255
    m = m[0]

    locs = get_mode_locations(y,m)

    #for loc in locs:
        #for i in range(loc[0],loc[1]):
    if len(locs):
        start = mode_start_mode(im,locs,m)
        extend_locs(im,locs,m,start)
        locs = get_line_locations(im,y,start,m)
        locs = [x for x in locs if (x[1] - x[0] > 3)]
        for loc in locs:
            for i in range(loc[0],loc[1]):
                newim[i,start:(start+m)] = 0
                im[i,start:(start+m)] = 255


        #for i in range(0, im.shape[0]):
            ## just check the outside
            #if (im[i,start-1] == 255) and (im[i,start+m] == 255):
                #if sum(im[i,start:(start+m)]) == 0:
                    #newim[i,start:(start+m)] = 0
                    #im[i,start:(start+m)] = 255

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


def extract_components(arrs):
    submaps = []
    for arr in arrs:
        img = arr['img']
        track_map = np.zeros(img.shape[:2],dtype=np.uint8)
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
    #squ = arr['contour'][:]
    squ = get_outer_rect(arr['img'],arr['contour'])
    #grow_rect_by_one(squ)

    outside = np.copy(arr['img'])

    cv2.fillPoly(outside, [squ], 255)

    outside = wrap_image(outside,arr)
    return outside

def separate_rectangles(inp):
    outsides = []
    for x in inp:
        outsides.append(separate_rectangle(x))
    return outsides


