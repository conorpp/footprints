import sys,os,json,argparse,time,math
from random import randint
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import numpy as np
from scipy import stats

from utils import *
from filters import *
from structures import wrap_image, Shape

from ocr import OCR_API

PARAMS = {'imageh':100,'imagew':100}

def init(x,orig=None):
    PARAMS['imageh'] = x.shape[0]
    PARAMS['imagew'] = x.shape[1]
    PARAMS['line-thickness'] = sample_line_thickness(x)
    print(x.shape, PARAMS['line-thickness'])
    if orig is not None:
        PARAMS['orig'] = np.copy(orig)


def grow_rect_top(c,square):
    # top side
    while still_inside(c, square[1], square[2]):
        square[1][1] -= 1
        square[2][1] -= 1
    square[1][1] += 1
    square[2][1] += 1

def grow_rect_bot(c,square):
    # bottom side
    while still_inside(c, square[3], square[4]):
        square[3][1] += 1
        square[4][1] += 1
        square[0][1] += 1
    square[3][1] -= 1
    square[4][1] -= 1
    square[0][1] -= 1

    # do left/right again

def grow_rect_right(c,square):
    # right side
    while still_inside(c, square[0], square[1]):
        square[0][0] += 1
        square[1][0] += 1
        square[4][0] += 1

    square[0][0] -= 1
    square[1][0] -= 1
    square[4][0] -= 1


def grow_rect_left(c,square):
    # left side
    while still_inside(c, square[2], square[3]):
        square[2][0] -= 1
        square[3][0] -= 1

    square[2][0] += 1
    square[3][0] += 1



def grow_rect(c,p):
    [x,y] = p

    square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    inc = 1

    # right side
    while still_inside(c, square[0], square[1]):
        square[0][0] += 1
        square[1][0] += 1
        square[4][0] += 1
        inc += 1

    dec = min(3,inc)
    square[0][0] -= dec
    square[1][0] -= dec
    square[4][0] -= dec
    inc = 1

    # left side
    while still_inside(c, square[2], square[3]):
        square[2][0] -= 1
        square[3][0] -= 1
        inc += 1

    dec = min(3,inc)
    square[2][0] += dec
    square[3][0] += dec

    grow_rect_top(c,square)
    grow_rect_bot(c,square)
    grow_rect_right(c,square)
    grow_rect_left(c,square)

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

    mat = np.copy(arr.img)
    mat, contours, hier = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #mat, contours, hier = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = arr.img

    if len(contours)>1:
        #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
        #try:
        output = cv2.connectedComponentsWithStats(tmp, 4)
        squares = []
        for x,y in output[3]:
            p = (int(x), int(y))
            if (cv2.pointPolygonTest(contours[1], p, 0) > 0 ):
                squ = grow_rect(contours[1],p)
                squares.append([squ, cv2.contourArea(squ), p])
        
        if len(squares):
            squares = sorted(squares, key = lambda x:x[1], reverse = True )
            square = squares[0]
            conf = rect_confidence(tmp, square[0])
            arr.conf = conf
            arr.center = square[2]
            arr.a1 = square[1]
            arr.contour = square[0]
            arr.area_ratio = square[1]/num_pixels

        arr.contour_area = cv2.contourArea(contours[1])
        arr.ocontour = contours[1]

        x,y,w,h = cv2.boundingRect(contours[1])
        arr.width = w
        arr.height = h
        arr.boundxy = (x,y)

    else: 
        print('warning, no contours')
        save_history(arr)
        raise ValueError('no contours')
        #sys.exit(1)


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


def grow_line(im,c,startp=None):
    if startp is None:
        x,y = centroid(c)
    else:
        x,y = startp
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

def analyze_line(spec):
    #try:
    c = spec.ocontour
    #except:
        #save_history(spec)
        #sys.exit(1)
    line,vertical = grow_line(spec['img'],c)
    #spec['vertical'] = vertical
    spec.vertical = 1 if (spec.height > spec.width) else 0
    spec.line = line
    spec.line_conf = line_confidence(spec.img,line)
    spec.line_length = math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1])
    spec.length_area_ratio = spec.line_length/spec.contour_area
    spec.aspect_ratio = spec.line_length/min([spec.width,spec.height])

    if spec.vertical:
        rowsum = scan_dim(spec.img,1)
    else:
        rowsum = scan_dim(spec.img,0)

    spec.sum['sum'] = rowsum
    #spec['colsum'] = {
        #'sum':colsum
        #}

    rowsum_trim = scan_trim(rowsum)
    #colsum = scan_trim(colsum)

    spec.sum['mode'] = stats.mode(rowsum_trim)

    spec.sum['distinct'] = len(np.unique(rowsum_trim))
    spec.sum['score'] = float(spec.sum['mode'][1])/len(rowsum_trim)

    return spec

def inherit_from_line(line,parent):
    for i in ('vertical','line','line_conf',
            'line_length','length_area_ratio',
            'aspect_ratio','sum'):
        setattr(line,i,getattr(parent,i))
        #line[i] = parent[i]


def analyze_lines(lines):
    for x in lines:
        specs = analyze_line(x)


def analyze_rectangles(rects):
    for im in rects:
        analyze_rectangle(im)

def analyze_triangles(rects,parentim):

    #if type(parentim) == type(Shape):
    if isinstance(parentim,Shape):
        parentim = parentim.img
    #else:
        #print(type(parentim))
        #print(type(Shape.__class__))

    for x in rects:
        area,tri = cv2.minEnclosingTriangle(x.ocontour)
        x.triangle = np.reshape(np.round(tri).astype(np.int32),(3,2))
        x.triangle_area = area
        inside = np.zeros(parentim.shape, dtype=np.uint8)
        cv2.drawContours(inside,[x.triangle],0,255,-1,offset=tuple(x.offset))
        inside = (cv2.bitwise_and(parentim, inside) + (inside != 255) * 255)

        #x['triangle-area-ratio'] = count_black(x['img'])/area
        x.triangle_area_ratio = count_black(inside)/area
        x.triangle_perimeter = cv2.arcLength(tri,True)

def analyze_ocr(inp):
    for x in inp:
        im = x.img
        OCR_API.SetImageBytes(im.tobytes(), im.shape[1], im.shape[0], 1, im.shape[1])
        text = OCR_API.GetUTF8Text()  # r == ri
        conf = OCR_API.MeanTextConf()
        x.ocr_conf = conf
        if text:
            symbol = text[0]
            x.symbol = symbol
        else:
            # check-periods
            x.symbol = None

def polish_rectangles(rects):
    for x in rects:
        ins = get_inner_rect(x.img,x.contour)
        out = get_outer_rect(x.img,x.contour)
        cen = get_center_rect(out,ins)
        x.rectangle = cen

def shift_line(im, pts,dim,perc,direc):
    blacks = (im == 0)
    udim = (dim + 1) & 1
    # right side
    while True:
        pixels = line_sum(blacks,pts)
        if pixels < (perc * abs(pts[0][udim] - pts[1][udim])):
            break
        pts[0][dim] += direc
        pts[1][dim] += direc

    return pts

def block_clipped_components(inp):
    good = []
    try:
        for x in inp:
            im = x.img
            if 0 not in im[:,0]:
                if 0 not in im[:,im.shape[1]-1]:
                    if 0 not in im[im.shape[0]-1,:]:
                        if 0 not in im[0,:]:
                            good.append(x)
    except:
        print('err')
        print(x.img)
        print(x.img.shape)
        sys.exit(1)

    return good


def get_inner_rect(im,c,confs=[.3,.3,.3,.3]):
    #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    square = np.copy(c[:])
    count = 25

    # right side
    square[0:2] = shift_line(im, square[0:2], 0, confs[0], -1)
    
    # top side
    square[1:3] = shift_line(im, square[1:3], 1, confs[1], 1)

    # left side
    square[2:4] = shift_line(im, square[2:4], 0, confs[2], 1)
    
    # bottom side
    square[4] = square[0]
    square[3:5] = shift_line(im, square[3:5], 1, confs[3], -1)
    square[0] = square[4]

    return square

def get_outer_rect(im,c,confs=[.3,.3,.3,.3]):
    #square = np.array([[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1],[x+1,y+1],])
    square = np.copy(c)
    count = 25

    zeros = np.zeros(im.shape)

    # right side
    square[0:2] = shift_line(im, square[0:2], 0, confs[0], 1)
    
    # top side
    square[1:3] = shift_line(im, square[1:3], 1, confs[1], -1)

    # left side
    square[2:4] = shift_line(im, square[2:4], 0, confs[2], -1)
    
    # bottom side
    square[4] = square[0]
    square[3:5] = shift_line(im, square[3:5], 1, confs[3], 1)
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

def circle_in_contour(c,p,r):
    total = 0
    x = p[0]
    y = p[1]
    pts = []
    for i in range(0,12):
        v = 2*math.pi * i/12.
        pt = (x + math.cos(v) * r,y + math.sin(v) * r)
        t = (cv2.pointPolygonTest(c, pt, 0) > 0 )
        pts.append(pt)
        if t:
            total += 1

    return total,pts

def analyze_circle(arr):
    squ = arr.contour
    c = arr.ocontour

    arr.circle = [(0,0),1]
    arr.circle_conf = 0

    if len(squ):
        x = int((squ[0][0] + squ[2][0])/2)
        y = int((squ[0][1] + squ[1][1])/2)
        if (cv2.pointPolygonTest(c, (x,y), 0) > 0 ):
            r = 1
            pts1,contour = circle_in_contour(c,(x,y),r)
            pts2 = 1
            while pts2:
                r += 1
                pts2,contour = circle_in_contour(c,(x,y),r)
                if pts2 < pts1:
                    break
            r -= 1
            cir = [(x,y),r]
            contour.append(contour[0])
            arr.circle = cir
            #ar'circle-contour'] = np.array(contour, dtype=np.uint8)
            arr.circle_contour = np.array(contour, dtype=np.int32)
            #print(arr['circle-contour'])
            arr.circle_conf = circle_confidence(arr.img,cir)

def analyze_circles(inp):
    for x in inp:
        analyze_circle(x)


def sample_line_thickness(arr):
    samples = []
    for col in range(0, arr.shape[1], 10):
        col = arr[:,col]

        locs_col = np.where(col == 0)[0]
        if len(locs_col):
            locs_col = np.split(locs_col, np.where(np.diff(locs_col) != 1)[0]+1)
            locs_col = [x.shape[0] for x in locs_col]

            samples += locs_col


    for row in range(0, arr.shape[0], 10):
        row = arr[row,:]

        locs_row = np.where(row == 0)[0]
        if len(locs_row):
            locs_row = np.split(locs_row, np.where(np.diff(locs_row) != 1)[0]+1)

            locs_row = [x.shape[0] for x in locs_row]

            samples += locs_row

    return stats.mode(samples)[0][0]


# grow a maximum circle starting from a line in a outer contour
def grow_semi_circle(outside,side,dim,direc):
    pt = [round((side[0][0]+side[1][0])/2), round((side[0][1]+side[1][1])/2)]

    while point_in_contour(outside,tuple(pt)):
        pt[dim] += direc

    pt[dim] -= direc*4

    r = 1
    total,circle = circle_in_contour(outside,pt,r)
    first_total = total
    while total == first_total:
        # grow the circle
        while total == first_total:
            total,circle = circle_in_contour(outside,pt,r)
            r += 1
        # move the circle
        pt[dim] -= direc*2
        total,circle = circle_in_contour(outside,pt,r)


    # take back the overstep
    pt[dim] += direc
    r -= 2
    pt = (int(round(pt[0])), int(round(pt[1])))

    return pt,r



def analyze_semi_rects(rects):
    #orig = color(polarize(np.copy(orig)))


    # right, top, left, bottom
    # dim, dir
    lut = [(0,1), (1,-1), (0,-1), (1,1)]

    for x in rects:
        semicircles = [((0,0),0,0)]*4

        # right, top, left, bottom
        for i,val in enumerate(x.conf):
            if val > .94: # TODO centralize this value
                continue
            rect = x.contour
            outside = x.ocontour
            side = rect[0+i:2+i]
            dim,direc = lut[i]
            
            pt,r = grow_semi_circle(outside,side,dim,direc)

            cconf = circle_confidence(x.img, (pt,r))

            # there's half a circle there, stop
            if cconf > .45:
                #print('circle conf',cconf)

                oppside = rect[(2+i)%4:(2+i)%4+2]
                oppdim = (dim+1)&1
                xleft =  pt[oppdim] - r+1
                xright = pt[oppdim] + r-1
                yval = pt[dim]


                oppside[0][oppdim] = xleft
                oppside[1][oppdim] = xright

                newside = [[xleft,yval],[xright,yval]]

                if i == 0:
                    newrect = [newside[1], newside[0], oppside[0], oppside[1], newside[1]]
                elif i == 1:
                    newrect = [oppside[1], newside[1], newside[0], oppside[0], oppside[1]]
                elif i == 2:
                    newrect = [oppside[1], oppside[0], newside[0], newside[1], oppside[1]]
                elif i == 3:
                    newrect = [newside[1], oppside[1], oppside[0], newside[0], newside[1]]

                rect = np.array(newrect)

                if i == 0:
                    grow_rect_top(outside,rect)
                    grow_rect_bot(outside,rect)
                elif i == 1:
                    grow_rect_left(outside,rect)
                    grow_rect_right(outside,rect)
                elif i == 2:
                    grow_rect_top(outside,rect)
                    grow_rect_bot(outside,rect)
                elif i == 3:
                    grow_rect_left(outside,rect)
                    grow_rect_right(outside,rect)

                x.contour = rect
                x.conf = rect_confidence(x.img, rect)
                #break

                semicircles[i] = (pt,r,cconf)
                #pt[0] += x['offset'][0]
                #pt[1] += x['offset'][1]
                #cv2.circle(orig,tuple(pt),r,(255,0x8c,0),1 )
                #pt[0] -= x['offset'][0]
                #pt[1] -= x['offset'][1]


        x.semi_circles = semicircles
        x.filled_rects = []
    #save(orig,'output2.png')


def make_irregular_shapes(shapes):
    for x in shapes:
        features = []
        for i in range(0,4):
            if x.conf[i] > .95:
                features.append(('line', x.contour[0+i:2+i], None, x.conf[i]))
            elif x['semi-circles'][i][2] > .45:
                circle = x.semi-circles[i][:2]
                degrees = [(270,90), (180,360), (90,270), (180,0)][i]
                features.append(('circle', circle, degrees, x.semi_circles[i][2]))
            else:
                raise ValueError('This shape is incomplete')

        x.features = features

def get_locations_of_value(im,line):
    sli = line_slice(im,line).flatten()
    locs = np.zeros(len(sli))
    for i,v in enumerate(sli):
        locs[i] = 255 == v
    #print(locs)
    locs = np.where(locs)[0]
    locs = np.split(locs, np.where(np.diff(locs) != 1)[0]+1)
    #print(locs)
    return locs



def fill_with_rects(rect):

    first_rect = rect.contour
    first_confs = rect.conf
    outside = rect.ocontour
    img = rect.img
    offset = rect.offset

    allrects = []
    gaps = [(first_rect,first_confs)]

    scratchpad = np.copy(img)

    while len(gaps):
        cur_rect,confs = gaps.pop()
        allrects.append(cur_rect)

        for i in range(0,4):


            if confs[i] > .95:
                continue

            side = cur_rect[0+i:2+i]
            dim = line_vert(side)
            xory = side[0][(dim+1)&1]
            #print('DIM:',side[0][dim])
            wht_offset = min(side[0][dim],side[1][dim])
            locs = get_locations_of_value(img,side)+wht_offset

            # BR, TR, TL, BL, BR
            # Set the old rectangle as black
            scratchpad[cur_rect[2][1]:cur_rect[0][1]+1, cur_rect[2][0]:cur_rect[0][0]+1] = 0

            for rg in locs:
                if len(rg) == 0:
                    continue
                center = rg[int(len(rg)/2)]
                if dim:
                    pt = (xory,center)
                else:
                    pt = (center,xory)

                rect2 = grow_rect(outside, pt)
                confs2 = rect_confidence(scratchpad,rect2)
                #print('---')
                #print(confs2)
                ##confs2[(i+2)%4] = 1 # clear the line we bridged from
                #print(confs2)
                #print('---')
                if np.sum(confs2>.95) >= 2:
                    gaps.append((rect2,confs2))


                ##
                #for wht in rg:
                    #if dim:
                        #pt = (xory,wht)
                    #else:
                        #pt = (wht,xory)

                    ###
                    #orig[pt[1]+offset[1], pt[0]+offset[0]] = [255,0,0]
                    ###
                #cv2.drawContours(orig,[rect2],0,[0,255,0],1, offset=tuple(offset))
                ##

    img = np.zeros(img.shape,dtype=np.uint8)+255
    for r in allrects:
        cv2.drawContours(img,[r],0,0,1)

    rect.filled_rects = allrects
    rect.scratch_pad = scratchpad

    _, contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[1]
    newc = [c[0][0]]
    lineset = []

    # remove little corner diagnols
    for i in range(0,len(c)-1):
        side = c[i:i+2]
        side = [side[0][0], side[1][0]]
        if line_len(side) > 2:
            lineset.append(side)
            
    lineset.append([c[-1][0], c[0][0]])
    lineset = np.array(lineset,dtype=np.int32)
    
    # contains constant dimension and the direction to move it toward center (0/1,1/-1)
    metaset = []

    # move each side inside by one
    for side in lineset:
        #print(side, line_len(side))
        dim = line_vert(side)
        odim = (dim+1)&1

        center = [(side[0][0]+side[1][0])/2, (side[0][1]+side[1][1])/2]
        center[odim] += 1


        if point_in_contour(c,tuple(center)):
            side[0][odim] += 1
            side[1][odim] += 1
            metaset.append((odim,1))
        else:
            side[0][odim] -= 1
            side[1][odim] -= 1
            metaset.append((odim,-1))

    features = []

    # calculate the features and their confidences
    for i,side in enumerate(lineset):
        conf = line_conf(img,side)
        #s,l = line_sum(img, side), line_len(side)
        #if l == 0:
            #conf = 0.0
        #else:
            #conf = 1.0 - float(s)/l

        features.append(('line', side, metaset[i], conf))

    rect.features = features

    ###
    #for line in lineset:
        #cv2.drawContours(orig,[line],0,[255,0,0],1, )
        #print('contour has %d sides' % ((len(newc)+1)/2))



def analyze_irregs(irregs):
    for x in irregs:
        fill_with_rects(x)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)
    import processors
    arr = load_image(sys.argv[1])
    if len(arr.shape) > 2:
        arr = polarize(arr)
    
    #arr[:,:120]=255
    #arr[:,-630:]=255
    #arr[:120,:]=255
    #arr[-360:,:]=255

    print(arr.shape)
    
    arr = wrap_image(arr)
    analyze_rectangles([arr])


    newlines,lineleftover = processors.find_line_features([arr])

    #arr =color(arr)
    #for i,insides in (outsides):
        #color = [randint(0,255) for x in range(0,3)]
        #cv2.drawContours(arr,[i],0,color,1)
        ##print(len(insides),'inside')
        #for j in insides:
            #cv2.drawContours(arr,[j],0,color,1)
    
    save(arr,'output.png')




