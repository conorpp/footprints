import sys,os,json,argparse,time,math
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import numpy as np
from scipy import stats

from utils import *
from filters import *

from ocr import OCR_API

def grow_rect(c):
    x,y = centroid(c)
    square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])

    # right side
    while still_inside(c, square[0], square[1]):
        square[0][0] += 1
        square[1][0] += 1
        square[4][0] += 1
    square[0][0] -= 1
    square[1][0] -= 1
    square[4][0] -= 1

    # top side
    while still_inside(c, square[1], square[2]):
        square[1][1] -= 1
        square[2][1] -= 1
    square[1][1] += 1
    square[2][1] += 1

    # left side
    while still_inside(c, square[2], square[3]):
        square[2][0] -= 1
        square[3][0] -= 1
    square[2][0] += 1
    square[3][0] += 1

    # bottom side
    while still_inside(c, square[3], square[4]):
        square[3][1] += 1
        square[4][1] += 1
    square[3][1] -= 1
    square[4][1] -= 1
    return square

def grow_rect_by_one(square):
    square[0][0] += 1
    square[1][0] += 1
    square[4][0] += 1

    square[1][1] -= 1
    square[2][1] -= 1

    square[2][0] -= 1
    square[3][0] -= 1

    square[3][1] += 1
    square[4][1] += 1


def analyze_rectangle(arr):

    # TODO
    num_pixels = float(960*760)

    mat = np.copy(arr['img'])
    mat, contours, hier = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = arr['img']

    if len(contours)>1:
        #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
        #try:
        square = grow_rect(contours[1])
        #except:
            #save_history(arr)
            #sys.exit(1)
        conf = rect_confidence(tmp, square)
        arr['conf'] = conf
        arr['a1'] = cv2.contourArea(square)
        arr['contour'] = square


        arr['area-ratio'] = cv2.contourArea(square)/num_pixels

        arr['contour-area'] = cv2.contourArea(contours[1])
        arr['ocontour'] = contours[1]

        x,y,w,h = cv2.boundingRect(contours[1])
        arr['width'] = w
        arr['height'] = h

    else: 
        print('warning, no contours')
        save_history(arr)
        sys.exit(1)

def move_point(im,p,i,di,expected):
    count = 0

    if di > 0:
        lim = im.shape[(i +1)&1]-1
    else:
        lim = 0

    for x in range(p[i],lim,di):
        #print (im[p[1],p[0]] )
        if im[p[1],p[0]] == expected:
            return count
        count += 1
        p[i] += di
    return count


def grow_line(im,c):
    x,y = centroid(c)
    p = [x,y]
    
    dx,dy = x,y
    dl,dr,dt,db = -1,-1,-1,-1

    # check if point is not on black
    if im[y,x] != 0:

        # move right
        r = move_point(im,[x,y],0,1,0)
        # move left
        l = move_point(im,[x,y],0,-1,0)
        # move up
        u = move_point(im,[x,y],1,-1,0)
        # move down
        d = move_point(im,[x,y],1,1,0)

        opts = []
        if r > 0: opts.append([r,0,1])
        if l > 0: opts.append([l,0,-1])
        if u > 0: opts.append([u,1,-1])
        if d > 0: opts.append([d,1,1])

        opts = sorted(opts, key = lambda x:x[0])
        p[opts[0][1]] += opts[0][2] * opts[0][0]

        # center it
        d = move_point(im,[p[0],p[1]], opts[0][1], opts[0][2],255)
        p[opts[0][1]] += int((d * opts[0][2] - opts[0][2])/2)

    
    # move it to farthest edge possible
    [x,y] = p
    # move right
    r = move_point(im,[x,y],0,1,255)
    # move left
    l = move_point(im,[x,y],0,-1,255)
    # move up
    u = move_point(im,[x,y],1,-1,255)
    # move down
    d = move_point(im,[x,y],1,1,255)

    opts = []
    if r > 0: opts.append([r,0,1])
    if l > 0: opts.append([l,0,-1])
    if u > 0: opts.append([u,1,-1])
    if d > 0: opts.append([d,1,1])

    opts = sorted(opts, key = lambda x:x[0],reverse=True)
    p[opts[0][1]] += opts[0][2] * opts[0][0] - opts[0][2]
    [x,y] = p

    # get opposite distance
    d = move_point(im,[x,y],opts[0][1], opts[0][2] * -1,255)-1
    if opts[0][1]:
        p2 = [x,y + d * opts[0][2] * -1]
    else:
        p2 = [x + d * opts[0][2] * -1,y]


    return np.array([(p[0],p[1]), (p2[0], p2[1])]), opts[0][1]

def line_confidence(im,c):
    s,_ = trace_sum(im,c)
    blackpixels = im.shape[0] * im.shape[1] - np.count_nonzero(im)
    return float(s)/blackpixels

def scan_dim(im,dim):
    return np.sum(im == 0,axis=dim)

def scan_trim(arr):
    return np.trim_zeros(arr)


def analyze_line(spec):
    #try:
    c = spec['ocontour']
    #except:
        #save_history(spec)
        #sys.exit(1)
    line,vertical = grow_line(spec['img'],c)
    #spec['vertical'] = vertical
    spec['vertical'] = 1 if (spec['height'] > spec['width']) else 0
    spec['line'] = line
    spec['line-conf'] = line_confidence(spec['img'],line)
    spec['line-length'] = math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1])
    spec['length-area-ratio'] = spec['line-length']/spec['contour-area']
    spec['aspect-ratio'] = spec['line-length']/min([spec['width'],spec['height']])

    if spec['vertical']:
        rowsum = scan_dim(spec['img'],1)
    else:
        rowsum = scan_dim(spec['img'],0)

    spec['sum'] = {
        'sum':rowsum
        }
    #spec['colsum'] = {
        #'sum':colsum
        #}

    rowsum_trim = scan_trim(rowsum)
    #colsum = scan_trim(colsum)

    spec['sum']['mode'] = stats.mode(rowsum_trim)
    spec['sum']['range'] = np.ptp(rowsum)
    spec['sum']['score'] = float(spec['sum']['mode'][1])/len(rowsum_trim)

    #spec['colsum']['mode'] = stats.mode(colsum)
    #spec['colsum']['range'] = np.ptp(colsum)
    #spec['colsum']['score'] = float(spec['colsum']['mode'][1])/len(colsum)
    return spec


def analyze_lines(lines):
    for x in lines:
        specs = analyze_line(x)

def analyze_rectangles(rects):
    for im in rects:
        analyze_rectangle(im)

def analyze_triangles(rects):
    for x in rects:
        area,tri = cv2.minEnclosingTriangle(x['ocontour'])
        x['triangle'] = np.round(tri).astype(np.int32)
        x['triangle-area'] = area
        x['triangle-area-ratio'] = count_black(x['img'])/area

def analyze_ocr(inp):
    for x in inp:
        im = x['img']
        OCR_API.SetImageBytes(im.tobytes(), im.shape[1], im.shape[0], 1, im.shape[1])
        text = OCR_API.GetUTF8Text()  # r == ri
        conf = OCR_API.MeanTextConf()
        x['ocr-conf'] = conf
        if text:
            symbol = text[0]
            x['symbol'] = symbol
        else:
            # check-periods
            if (x['width'] == x['height']) and (count_black(im) < 17) and x['width']<8:
                x['symbol'] = '.'
                x['ocr-conf'] = 75
            else:
                x['symbol'] = None



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    if len(arr.shape) > 2:
        arr = polarize(arr)
    print(arr.shape)
    arr = wrap_image(arr)
    
    analyze_rectangle(arr)

    retval,triangle = cv2.minEnclosingTriangle(arr['ocontour'])
    print(triangle)
    print(retval)
    tri2 = triangle

    tri2 = np.array(tri2)
    print ('triangle area exa',cv2.contourArea(tri2))
    tri2 = np.round(tri2).astype(np.int32)
    print ('triangle area aprx',cv2.contourArea(tri2))
    print('black pixels: ',count_black(arr['img']))


    arr['img'] = color(arr['img'])
    cv2.drawContours(arr['img'],[tri2],0,[255,0,0],1)
    save(arr['img'],'output.png')




