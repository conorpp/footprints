import sys,os,json,argparse,time,math
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import numpy as np
from scipy import stats

from image_proc import *
from plotim import *

def trim_image(arr):
    arr['img'],x,y = trim(arr['img'])
    arr['offset'][0] += x
    arr['offset'][1] += y

def trim_images(imgs):
    for x in imgs:
        trim_image(x)

def tag_rectangle(arr):

    # TODO
    num_pixels = float(960*760)

    mat = np.copy(arr['img'])
    mat, contours, hier = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = arr['img']

    if len(contours)>1:
        #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
        square = grow_rect(contours[1])
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
    try:
        x,y = centroid(c)
    except:
        show(im)
        while True:pass
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

def tag_line(spec):
    c = spec['ocontour']
    line,vertical = grow_line(spec['img'],c)
    spec['vertical'] = vertical
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

    rowsum = scan_trim(rowsum)
    #colsum = scan_trim(colsum)

    spec['sum']['mode'] = stats.mode(rowsum)
    spec['sum']['range'] = np.ptp(rowsum)
    spec['sum']['score'] = float(spec['sum']['mode'][1])/len(rowsum)

    #spec['colsum']['mode'] = stats.mode(colsum)
    #spec['colsum']['range'] = np.ptp(colsum)
    #spec['colsum']['score'] = float(spec['colsum']['mode'][1])/len(colsum)
    return spec


def get_lines(lines):
    res = []
    for x in lines:
        specs = tag_line(x)
        res.append(specs)
    return res

def get_rectangles(rects):
    for im in rects:
        tag_rectangle(im)
    return rects

def filter_rectangles(rects):
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

def filter_lines(rects):
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
def filter_fresh(fresh):
    good = []
    for x in fresh:
        nz = np.count_nonzero(x['img'])
        z = x['img'].shape[0] * x['img'].shape[1] - nz
        if z > 1:
            good.append(x)
    return good


def get_num_from_name(n):
    nums = '0123456789'
    num = ''
    for x in n:
        if x in nums:
            num += x
    return int(num)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <inputs.png [...]> <out-dir>' % sys.argv[0])
        sys.exit(1)


    names = sys.argv[1:-1]
    output = sys.argv[-1]

    #images = [load_image(x) for x in names]
    res = []
    for i,n in enumerate(names):
        im = load_image(n)

        t1 = int(round(time.time() * 1000))

        specs = tag_rectangle(im)
        specs['i'] = i
        specs['name'] = n
        res.append(specs)


        t2 = int(round(time.time() * 1000))
        print('get_rect: %d' % (t2-t1))

        im = color(specs['im'])
        cv2.drawContours(im,[specs['rect']],0,[255,0,0],1)

        Image.fromarray(im).save(output+'/'+('cat-%d.png'%get_num_from_name(n)))


    res = sorted(res, key=lambda x: x['area-ratio'])
    for x in res:
        print('i: %d, name: %s, conf: %.2f, ar: %.3f a1: %.2f a2: %.2f' % (x['i'], x['name'], 
                    x['conf'], x['area-ratio'],x['a1'],x['a2']))

    print(' '.join([x['name'] for x in res]))


