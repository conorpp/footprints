import sys,os,json,argparse,time,math
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import numpy as np
from scipy import stats

from utils import *
from filters import *

from ocr import OCR_API

PARAMS = {'imageh':100,'imagew':100}

def init(x):
    PARAMS['imageh'] = x['img'].shape[0]
    PARAMS['imagew'] = x['img'].shape[1]
    print(PARAMS)

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
        square[0][1] += 1
    square[3][1] -= 1
    square[4][1] -= 1
    square[0][1] -= 1
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

    num_pixels = float(PARAMS['imageh']*PARAMS['imagew'])

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

    spec['sum']['distinct'] = len(np.unique(rowsum_trim))
    spec['sum']['score'] = float(spec['sum']['mode'][1])/len(rowsum_trim)

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
        x['triangle-perimeter'] = cv2.arcLength(tri,True)

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
            #if (x['width'] == x['height']) and (count_black(im) < 17) and x['width']<8:
            if float(count_black(im))/(max(x['width'],x['height'])**2) > .275:
                #print(float(count_black(im))/(max(x['width'],x['height'])**2))
                x['symbol'] = '.'
                x['ocr-conf'] = 75
            else:
                x['symbol'] = None

def polish_rectangles(rects):
    for x in rects:
        ins = get_inner_rect(x['img'],x['contour'])
        out = get_outer_rect(x['img'],x['contour'])
        cen = get_center_rect(out,ins)
        x['rectangle'] = cen

def shift_line(im, pts,dim,perc,direc):
    _zeros = np.zeros(im.shape)
    udim = (dim + 1) & 1
    # right side
    while True:
        _zeros[:] = 1
        cv2.drawContours(_zeros,[pts],0,0,1)
        pixels = np.sum(_zeros==im)
        if pixels < (perc * abs(pts[0][udim] - pts[1][udim])):
            break
        pts[0][dim] += direc
        pts[1][dim] += direc

    return pts

def block_clipped_components(inp):
    good = []
    for x in inp:
        im = x['img']
        if 0 not in im[:,0]:
            if 0 not in im[:,im.shape[1]-1]:
                if 0 not in im[im.shape[0]-1,:]:
                    if 0 not in im[0,:]:
                        good.append(x)

    return good


def get_inner_rect(im,c):
    #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    square = np.copy(c[:])
    count = 25

    # right side
    square[0:2] = shift_line(im, square[0:2], 0, .9, -1)
    
    # top side
    square[1:3] = shift_line(im, square[1:3], 1, .9, 1)

    # left side
    square[2:4] = shift_line(im, square[2:4], 0, .9, 1)
    
    # bottom side
    square[4] = square[0]
    square[3:5] = shift_line(im, square[3:5], 1, .9, -1)
    square[0] = square[4]

    return square

def get_outer_rect(im,c):
    #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    square = np.copy(c)
    count = 25

    zeros = np.zeros(im.shape)

    # right side
    square[0:2] = shift_line(im, square[0:2], 0, .9, 1)
    
    # top side
    square[1:3] = shift_line(im, square[1:3], 1, .9, -1)

    # left side
    square[2:4] = shift_line(im, square[2:4], 0, .9, -1)
    
    # bottom side
    square[4] = square[0]
    square[3:5] = shift_line(im, square[3:5], 1, .9, 1)
    square[0] = square[4]


    return square

def get_center_rect(out,ins):
    cen = np.copy(out)
    cen[0][0] -= (out[0][0] - ins[0][0])/2.
    cen[0][1] -= (out[0][1] - ins[0][1])/2.

    cen[1][0] -= (out[1][0] - ins[1][0])/2.
    cen[1][1] += (-out[1][1]+ ins[1][1])/2.

    cen[2][0] += (-out[2][0]+ ins[2][0])/2.
    cen[2][1] += (-out[2][1]+ ins[2][1])/2.

    cen[3][0] += (-out[3][0]+ ins[3][0])/2.
    cen[3][1] -= (out[3][1] - ins[3][1])/2.

    cen[4] = cen[0]

    return cen

#def get_flat_locations(y,flat_size):
    #locs = []
    #start = None
    #end = None
    #lastp = -1
    #print('y len',len(y))
    #for i,p in enumerate(y):
        #if p == lastp:
            #if start is None:
                #start = i
                #end = i+1
            #else:
                #end += 1
        #else:
            #if start is not None and ((end-start) >= flat_size):
                #locs.append([start,end])
            #start = None
            #if p:
                #lastp = p
    #return locs

def sum_crossings(im,dim):
    if dim == 0:
        im = np.transpose(im)

    sums = np.zeros(im.shape[0])
    indexs = np.zeros(im.shape[0])
    lastp = 255 # white

    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            if im[i,j] != lastp:
                sums[i] += 1
                lastp = im[i,j]

                indexs[i] = j

    if dim == 0:
        im = np.transpose(im)

    return sums/2, indexs

def trim_crossings(sums):
    #sums = np.trim_zeros(sums)
    sums = sums + (sums == 0)
    for i,p in enumerate(sums):
        if p == 1:
            sums[i] = 0
        else:
            break

    for i in range(len(sums)-1,-1,-1):
        if sums[i] == 1:
            sums[i] = 0
        else:
            break

    #sums = np.trim_zeros(sums)
    return sums

def center_locs(locs):
    return [int(x[0] + (x[1] - x[0])/2) for x in locs]

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

def cut_linking_lines(arrs):
    for arr in arrs:
        img = arr['img']
        cut_linking_line(img)

def cut_linking_line(arr):
    #TODO derive these from something
    perpendicular_line_length = 10
    min_line_length = 10
    variation = 3

    lines,indexs = sum_crossings(arr,0)
    sums = scan_dim(arr,0)
    lines = lines + (sums > perpendicular_line_length )

    lines = trim_crossings(lines)
    locs = get_mode_locations(lines,1)
    locs = [x for x in locs if ((x[1] - x[0]) >= min_line_length)]

    locs = [x for x in locs if (len(np.unique( indexs[x[0]:x[1]] )) < variation)]

    col_cut_points = center_locs(locs)
    for x in col_cut_points:
        if (arr.shape[0] - np.sum(arr[:,x])/255) < 5:
            arr[:,x] = 255


    lines,indexs = sum_crossings(arr,1)
    sums = scan_dim(arr,1)
    lines = lines + (sums > perpendicular_line_length )
    lines = trim_crossings(lines)

    #plt.plot(lines)
    #plt.show()

    locs = get_mode_locations(lines,1)

    locs2 = []
    for x in locs:
        l = x[1] - x[0]
        if l < min_line_length:
            continue

        # look at center min_line_length points
        off = int(l/2)
        min_line_h = int(min_line_length/2)
        uni = np.unique( indexs[x[0] + off - min_line_h:x[1] - off + min_line_h] )
        if len(uni) < variation:
            locs2.append(x)

    locs = locs2
    #locs = [x for x in locs if (len(np.unique( indexs[x[0]:x[1]] )) < variation)]
    row_cut_points = center_locs(locs)
    for x in row_cut_points:
        if (arr.shape[1] - np.sum(arr[x,:])/255) < 5:
            arr[x,:] = 255



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    if len(arr.shape) > 2:
        arr = polarize(arr)

    print(arr.shape)
    #rowsum = scan_dim(arr,0)
    #colsum = scan_dim(arr,1)
    
    #cut_linking_line(arr)


    arr = wrap_image(arr)

    analyze_rectangle(arr)

    ##cv2.drawContours(arr['img'],[arr['contour'][0:2]],0,[0,255,255],1)
    ##cv2.drawContours(arr['img'],[arr['contour'][1:3]],0,[0,255,0],1)
    ##cv2.drawContours(arr['img'],[arr['contour'][2:4]],0,[255,0,0],1)
    ##cv2.drawContours(arr['img'],[arr['contour'][3:5]],0,[255,0,255],1)
    
    #outer_sq = get_outer_rect(arr['img'],arr['contour'])
    #inner_sq = get_inner_rect(arr['img'],arr['contour'])
    #center_sq = get_center_rect(outer_sq,inner_sq)

    #print('out:',outer_sq)
    #print('cen:',center_sq)
    #print('in:',inner_sq)

    #cv2.drawContours(arr['img'],[inner_sq],0,[255,255,0],1)
    #cv2.drawContours(arr['img'],[outer_sq],0,[0,0,255],1)
    
    #cv2.fillPoly(arr['img'], [inner_sq], [255]*3)
    arr['img'] = color(arr['img'])
    cv2.drawContours(arr['img'],[arr['contour']],0,[255,0,0],1)

    save(arr,'output.png')




