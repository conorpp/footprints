import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2

from image_proc import *
from detect_corners import *
from plotim import *

# x: row, y: column, z:rgba, origin is upper-left

# polarize
def polarize(arr):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    (thresh, arr) = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return arr

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

def apply_mapping(im,mapping):
    cpy = np.copy(im)
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            if mapping[i,j]:
                cpy[i,j,0] = 255
                cpy[i,j,1] = 1
                cpy[i,j,2] = 1
    return cpy

def mapping2greyscale(mapping):
    mapping = np.array((mapping == 0) * 255, dtype=np.uint8)
    return mapping

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
                    submap = mapping2greyscale(submap)
                    submap = wrap_image(submap)
                    submaps.append( submap )


    return submaps

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    arr = remove_alpha(arr)

    orig = np.copy(arr)
    arr = polarize(arr)
    arr = wrap_image(arr)

    submaps = extract_components(arr)
    snapshot_imgs(submaps,'extract_components parent',arr)
    snapshot_imgs(submaps,'before trim')

    trim_images(submaps)
    snapshot_imgs(submaps,'after trim')

    rectangles = get_rectangles(submaps)
    #snapshot_imgs(rectangles,'after get_rectangles')

    rect_f,leftover = filter_rectangles(rectangles)
    #snapshot_imgs(rect_f,'im a rectangle')
    #snapshot_imgs(leftover,'im not a rectangle')

    lines = get_lines(leftover)

    #snapshot_imgs(lines,'after get_lines')

    lines,leftover = filter_lines(lines)
    
    #snapshot_imgs(lines,'im a line')
    #snapshot_imgs(leftover,'im not a line')

    #fresh,leftover = break_lines(leftover)
    #it = 0
    #print('fresh')
    #while len(fresh):
        #it += 1
        #submaps = []
        #for x in fresh:
            #subm = extract_components(x)
            #submaps += subm
        #submaps = filter_fresh(submaps)
        #trim_images(submaps)
        #rectangles = get_rectangles(submaps)
        ##leftover3,leftover2 = filter_rectangles(rectangles)
        #lines2 = get_lines(rectangles)
        #lines2,leftover2 = filter_lines(lines2)
        #print('pass %d.  found %d lines and %d more leftover' % (it,len(lines2), len(leftover2)))
        
        #fresh,leftover2 = break_lines(leftover2)
        #print('there\'s %d possible lines and %d not containing lines' % (len(fresh), len(leftover2)))
        ##if it == 1:
            ##for i,x in enumerate(fresh):
                ##save(x['orig'],'output%d.png' % i)
            ##sys.exit(1)
        #break
        #lines += lines2
        #leftover += leftover2   # not lines
        #break

    #lines += lines2
    #print('%d classified lines.  %d from second pass' % (len(lines),len(lines2)))
    #leftover += leftover2
    #print('%d unclassified items. %d from second pass' % (len(leftover), len(leftover2)))

    for x in (lines + rect_f):
        x['cl'] = True
    for x in (leftover):
        x['cl'] = False
    if 0:
        for x in sorted(lines+leftover, key=lambda x: x['id']):
            print_img(x)
            cpy = np.copy(orig)
            cv2.drawContours(cpy,[x['line']],0,[255,0,0],1,offset=tuple(x['offset']))
            encircle(cpy, x['line'], offset=x['offset'])

            xx,yy,w,h = cv2.boundingRect(x['ocontour'])
            [xx,yy] = xx+x['offset'][0],yy+x['offset'][1]
            cv2.rectangle(cpy,(xx,yy),(xx+w,yy+h),(0,0,255),2)
            postfix = 'C' if x['cl'] else 'U'
            save(cpy,'out/line%c%d.png' % (postfix,x['id']))
            save(x['img'],'out/item%c%d.png' % (postfix,x['id']))


    for x in rect_f:
        cv2.drawContours(orig,[x['contour']],0,[255,0,255],1, offset=tuple(x['offset']))
    for x in lines:
        #cv2.drawContours(orig,[x['contour']],0,[0,0,255],1, offset=x['offset'])
        #cv2.drawContours(orig,[x['ocontour']],0,[0,255,0],1, offset=x['offset'])
        #for j,i in x['line']:
            #print ('line',i,j)
            #orig[i+x['offset'][1],j+x['offset'][0],0] = 255
            #orig[i+x['offset'][1],j+x['offset'][0],1] = 0
            #orig[i+x['offset'][1],j+x['offset'][0],2] = 0
        cv2.drawContours(orig,[x['line']],0,[0,128,0],2, offset=tuple(x['offset']))

    for x in leftover:
        if contains_line(x):
            cv2.drawContours(orig,[x['ocontour']],0,[255,255,0],1, offset=tuple(x['offset']))
        else:
            cv2.drawContours(orig,[x['ocontour']],0,[255,0,0],1, offset=tuple(x['offset']))

    save(orig,'output.png')
    save_history(rect_f[1])

    #for x in rectangles:
        #print(x)

    #for i,x in enumerate(submaps):
        #print(i)
        #out = gen_image(x)
        
        #out = Image.fromarray(out)
        #out.save(output + ('/item%d.png' % i))





