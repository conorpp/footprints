import sys,os,json,argparse,time
import cv2
import matplotlib as plt
from PIL import Image, ImageDraw

import numpy as np

from image_proc import *

def area_ratio(shape, c):
    return cv2.contourArea(c)/float(shape[0] * shape[1])

def tag_rectangle(arr):

    orig_shape = arr.shape

    arr,x,y = trim(greyscale(arr))
    
    specs = {'conf':0, 'area-ratio':0,'a1':0, 'a2':0,'rect':0,
            'offset':(x,y)}

    output = sys.argv[2]


    mat = np.copy(arr)
    
    mat, contours, hier = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #arr = color(arr)
    tmp = np.copy(arr)

    if len(contours)>1:
        square = grow_rect(contours[1])
        conf = rect_confidence(tmp, square)
        specs['conf'] = conf
        specs['area-ratio'] = cv2.contourArea(square)/float(orig_shape[0] * orig_shape[1])
        specs['a1'] = cv2.contourArea(square)
        specs['a2'] = float(orig_shape[0] * orig_shape[1])
        specs['rect'] = square
    else: 
        print('warning, contours')


    specs['im'] = tmp
    return specs

def get_rectangles(rects):
    res = []
    for im in rects:
        specs = tag_rectangle(np.copy(im))
        res.append(specs)
    return res

def filter_rectangles(rects):
    filtered = []
    for x in rects:
        if x['area-ratio'] > 0.001:
            if x['conf'] > .95:
                filtered.append(x)
    return filtered

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





