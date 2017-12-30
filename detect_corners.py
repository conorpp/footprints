import sys,os,json,argparse
import cv,cv2
import matplotlib as plt
from PIL import Image, ImageDraw

import numpy as np


if len(sys.argv) != 3:
    print 'usage: %s <input.png> <output>' % sys.argv[0]
    sys.exit(1)

im = Image.open(sys.argv[1])
output = sys.argv[2]


arr = np.array(im)
if arr.shape[2] == 4:
    arr = np.delete(arr, 3, 2) # get rid of alpha..

# get rid of extra channels
arr = np.delete(arr, 2, 2)
arr = np.delete(arr, 1, 2)
#arr = arr.reshape(arr.shape[0],arr.shape[1])


def trim(im):
    padding = 2
    colsum = im.shape[0] * 255 * 1
    # trim columns left
    while sum(im[:,padding].flatten()) == colsum:
        im = np.delete(im, 0, 1)

    # trim columns right
    while sum(im[:,-padding-1].flatten()) == colsum:
        im = np.delete(im, -1, 1)

    rowsum = im.shape[1] * 255 * 1

    # trim rows top
    while sum(im[padding].flatten()) == rowsum:
        im = np.delete(im, 0, 0)

    # trim rows bottom
    while sum(im[-padding-1].flatten()) == rowsum:
        im = np.delete(im, -1, 0)

    return im


out = arr
out = trim(arr)
h,w,z = out.shape

#mat = cv.fromarray(out)
mat = np.copy(out)
contours, hierarchy = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                #img, countours, index/all, color, thickness

print len(contours)
out = np.tile(out, (1,1,3))

for i in range(1,1+1):
    tmp = np.copy(out)
    #cv2.drawContours(tmp, contours, i, (0,255,0), 1)

    #shape = cv2.approxPolyDP(contours[i], 60, 1)
    #cv2.drawContours(tmp, [shape], 0, (255,0,0), 1)
    moms = cv2.moments(contours[i])
    x = int(moms['m10']/moms['m00'])
    y = int(moms['m01']/moms['m00'])
    
    square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    def still_inside(c,p1,p2):
        p1 = (p1[0],p1[1])
        p2 = (p2[0],p2[1])
        return (cv2.pointPolygonTest(c, p1,0) > 0 ) and (cv2.pointPolygonTest(c, p2,0) > 0 )

    # right side
    while still_inside(contours[i], square[0], square[1]):
        square[0][0] += 1
        square[1][0] += 1
        square[4][0] += 1
    square[0][0] -= 1
    square[1][0] -= 1
    square[4][0] -= 1

    # top side
    while still_inside(contours[i], square[1], square[2]):
        square[1][1] -= 1
        square[2][1] -= 1
    square[1][1] += 1
    square[2][1] += 1

    # left side
    while still_inside(contours[i], square[2], square[3]):
        square[2][0] -= 1
        square[3][0] -= 1
    square[2][0] += 1
    square[3][0] += 1

    # bottom side
    while still_inside(contours[i], square[3], square[4]):
        square[3][1] += 1
        square[4][1] += 1
    square[3][1] -= 1
    square[4][1] -= 1


    cv2.drawContours(tmp, [square], 0, (255,0,0), 1)

    tmp = Image.fromarray(tmp)
    print i
    tmp.save(output + ('/contour%d.png' % i))

