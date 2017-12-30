
from PIL import Image, ImageDraw
import cv,cv2
import numpy as np

def greyscale(im):
    if im.shape[2] == 4:
        im = np.delete(im, 3, 2)
    if im.shape[2] == 3:
        im = np.delete(im, 2, 2)
    if im.shape[2] == 2:
        im = np.delete(im, 1, 2)
    return im

def color(arr):
    if arr.shape[2] != 1: raise ValueError('needs to be single channel')
    arr = np.tile(arr, (1,1,3))
    return arr

def load_image(name):
    im = Image.open(name)
    return np.array(im)


def trim(im):
    padding = 2
    colsum = im.shape[0] * 255 * 1

    # trim columns left
    jl= 0
    while 0 not in im[:,jl+padding].flatten():
        jl+= 1
        #im = np.delete(im, 0, 1)
    
    # trim columns right
    jr = im.shape[1]-1
    while 0 not in im[:,jr - padding-1].flatten():
        jr -= 1
    
    im = im[:,jl:jr]
    rowsum = im.shape[1] * 255 * 1

    # trim rows top
    it = 0
    while 0 not in (im[padding + it].flatten()):
        it += 1

    ib = im.shape[0] - 1
    # trim rows bottom
    while 0 not in (im[ib - padding -1].flatten()):
        ib -= 1

    im = im[it:ib,:]

    return im

#def preprocess(im):

def still_inside(c,p1,p2):
    p1 = (p1[0],p1[1])
    p2 = (p2[0],p2[1])
    return (cv2.pointPolygonTest(c, p1,0) > 0 ) and (cv2.pointPolygonTest(c, p2,0) > 0 )

def grow_rect(c):
    moms = cv2.moments(c)
    x = int(moms['m10']/moms['m00'])
    y = int(moms['m01']/moms['m00'])
    
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

def show(im):
    tmp = Image.fromarray(im)
    tmp.show()

def save(nparr,name):
    im = Image.fromarray(nparr)
    im.save(name)

def trace_sum(im,contour):

    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.drawContours(mask,[contour],0,255,1)
    pixelpoints = np.transpose(np.nonzero(mask))

    total = 0
    for i,j in pixelpoints:
        total += (sum(im[i,j]) == 0)

    return total, len(pixelpoints)

def rect_confidence(im,con):
    s,t = trace_sum(im,con)
    return float(s)/t


