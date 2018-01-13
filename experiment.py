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

def timestamp(): return int(round(time.time() * 1000))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    arr = remove_alpha(arr)


    #arr[:,320:] = 255
    #arr[:200,:] = 255
    #arr[304:,:] = 255
    #arr[:190,:] = 255
    #arr[:,:250] = 255

    orig = np.copy(arr)
    arr = polarize(arr)
    im = np.copy(arr)

    t1 = timestamp()
    line_thickness = sample_line_thickness(arr)
    t2 = timestamp()
    print('line sampling time: %d ms' % (t2-t1))
    print('line thickness: %d pixels' % (line_thickness))

    t1 = timestamp()
    xlocs, ylocs = get_intersects(arr, (im.shape[0]*.03), (im.shape[1]*.03))
    t2 = timestamp()
    print('filter time: %d ms' % (t2-t1))




    corner_map = get_corner_map(arr,xlocs,ylocs)


    # dual of corner map
    line_map_h, line_map_v = get_line_map(arr,corner_map,xlocs,ylocs)

    # intersects over black pixel
    tlcorners, trcorners, brcorners, blcorners = get_corners(line_map_h,line_map_v,xlocs,ylocs)


    # detect the rectangles
    rectangles = get_rectangles(arr,tlcorners, trcorners, brcorners, blcorners)




    corners = blcorners + brcorners + tlcorners + trcorners
    corners = np.array(corners)

    # block redundant rectangles and detect overlapping rects
    

    rejects = []
    last_len = -1
    rectangles = [np.array(x) for x in rectangles]
    rectangles = block_small_rects(arr,rectangles)
    rectangles = sorted(rectangles,key = lambda x: rect_area(x))
    print(len(rectangles),'rectangles before')
    rectangles = remove_super_rects(arr,rectangles)
    print(len(rectangles),'rectangles after')

    #over_lap = detect_overlap(arr,rectangles)
    islands = []
    while len(rectangles):
        r = rectangles.pop(0)
        adj_r = detect_overlap(arr,rectangles,r)

        if adj_r is None:
            islands.append(r)
            continue
        print('reconciling..')
        rejects += reconcile_overlaps(arr,rectangles,(r,adj_r))

    rectangles = islands

    rectangles, bad= remove_side_rects(arr,rectangles)
    rejects += bad

    groups = group_rects(rectangles, line_thickness * 6)
    rectangles = []
    print('%d groups' % len(groups))
    for xdim,ydim,gro in groups:
        print('  group of %d rects: %dx%d' % (len(gro), xdim,ydim))
        rectangles += gro

    for i,x in enumerate(xlocs):
        orig[:,x] = [255,255,0]

    for i,x in enumerate(ylocs):
        orig[x,:] = [255,255,0]


    #for x,y in intersects:
        #orig[y,x] = [0,0,255]

    #for p in potential_corners:
        #cv2.circle(orig,p,10,(255,0x8c,0),2 )
    #for l in potential_lines:
        #cv2.line(orig,tuple(l[0]),tuple(l[1]), (255,200,0),1)

    #print(len(lines),' solid lines')
    #for l in lines:
        #cv2.line(orig,tuple(l[0]),tuple(l[1]), (0,255,0),1)
    
    #print(len(corners),'solid corners')
    #for c in corners:
        #cv2.line(orig,tuple(c[0]),tuple(c[1]), (0,255,255),1)
        #cv2.line(orig,tuple(c[1]),tuple(c[2]), (0,255,255),1)
        #cv2.circle(orig,tuple(c[1]),10,(255,0,255),2 )


    print(len(rectangles),'rectangles')
    for i,r in enumerate(rectangles):
        r = np.array(r)
        #if i == 3:
        cv2.drawContours(orig,[r],0,[255,0,255],2)
        #print(r)
    #for i,r in enumerate(blocked_rects):
        #if i == 1:
            #cv2.drawContours(orig,[r],0,[0,0,255],1)


    for i,r in enumerate(rejects):
        cv2.drawContours(orig,[np.array(r)],0,[255,0,0],2)
        #print('rejct',r)
        #cv2.drawContours(orig,[np.array(r[1])],0,[128,0,255],2)

    save(orig,'output.png')
    #for x in sorted(leftover + rectangles, key = lambda x:x['id']):

        #if x['id'] == 497:
        #if x in triangles:
            #save_history(x)
        #if x in rectangles:
            #save_history(x)
        #print('saving %d' % (x['id'],) )
        #x['img'] = color(x['img'])
        #cv2.drawContours(x['img'],[x['ocontour']],0,[255,0,0],1, offset=tuple(x['offset']))
        #save(x,'out/item%d.png' % x['id'])
        #pass




