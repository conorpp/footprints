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
                        #starts.append(s)
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
            white_both_sides = (im[loc[1],start-1] == 255) 
            white_both_sides = white_both_sides and (im[loc[1],start+m] == 255)
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

faulty_id = 0
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
        if start > -1:
            extend_locs(im,locs,m,start)
            locs = get_line_locations(im,y,start,m)
            locs = [x for x in locs if ((x[1] - x[0]) > 2)]
            for loc in locs:
                for i in range(loc[0],loc[1]):
                    newim[i,start:(start+m)] = 0
                    im[i,start:(start+m)] = 255
        else:
            global faulty_id
            faulty_id +=1
            print('faulty image',faulty_id)
            #save(im,'out/fault%d.png'%faulty_id)


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
            #if (i+1) <= imax:
                #checkout(i+1,j+1)

            # top-right
            #if (i-1) >= 0:
                #checkout(i-1,j+1)

        # left
        if (j-1) >= 0:
            checkout(i,j-1)

            # bottom-right
            #if (i+1) <= imax:
                #checkout(i+1,j-1)

            # top-right
            #if (i-1) >= 0:
                #checkout(i-1,j-1)

        # top
        if (i - 1) >= 0:
            checkout(i-1,j)

        # bottom
        if (i + 1) <= imax:
            checkout(i+1,j)

    return trackmap

def extract_features(arrs):
    print('extracting..')
    coms = extract_components(arrs)
    #trim_images(coms)

    cut_linking_lines(coms)
    coms2 = extract_components(coms)
    #trim_images(coms2)
    #while len(coms2) > len(coms):
        #print('coms: %d, coms2: %d'%(len(coms),len(coms2)))
        #coms = coms2
        #cut_linking_lines(coms)
        #coms2 = extract_components(coms)
        #trim_images(coms2)

    print('coms2: %d'%(len(coms2)))
    return coms2


def extract_components(arrs):
    submaps = []
    for arr in arrs:
        img = arr['img']
        #track_map = np.zeros(img.shape[:2],dtype=np.uint8)
        _submaps = get_isolated_images(img)
        submaps += [wrap_image(x[0],arr,x[1]) for x in _submaps]
        #for i in range(0,img.shape[0]):
            #for j in range(0,img.shape[1]):
                #if img[i][j] == 0:
                    #if not track_map[i,j]:
                        #track_map[i,j] = 1
                        #submap = explore(img,i,j)
                        #track_map += submap
                        #def mapping2greyscale(mapping):
                            #mapping = np.array((mapping == 0) * 255, dtype=np.uint8)
                            #return mapping
                        #submap = mapping2greyscale(submap)
                        #submap = wrap_image(submap,arr)
                        #submaps.append( submap )

    return submaps

# all black connected components
def get_isolated_contours(arr):
    try:
        wtfwhy = np.array(np.copy(arr).tolist(),dtype=np.uint8)
        mat, contours, hier = cv2.findContours(wtfwhy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        print('exception',e)
        print(arr.shape)
        print(arr.dtype)
        print(arr[:,arr.shape[1]-3])
        save(arr,'output.png')
        sys.exit(1)
    #for i,x in enumerate(hier[0]):
        #print(i,x)
    if len(contours) < 2:
        return []
    new_starts= [1]
    outsides = []
    while len(new_starts):
        i = new_starts.pop(0)
        new_outsides = []
        while i > -1:
            new_outsides.append([i,[]])
            i = hier[0][i][0]

        for out in new_outsides:
            i = hier[0][out[0]][2]
            if i > -1:
                while i > -1:
                    out[1].append(i)
                    if hier[0][i][2] > -1:
                        new_starts.append(hier[0][i][2])
                    i = hier[0][i][0]

        outsides += new_outsides
    ret = []
    for out,insides in outsides:
        oc = contours[out]
        ins = [contours[i] for i in insides]
        ret.append([oc,ins])

    return ret

# return [[im, offset from parent],...]
def get_isolated_images(arr):
    iso = []
    outsides = get_isolated_contours(arr)

    for i,(c,insides) in enumerate(outsides):
        x,y,w,h = cv2.boundingRect(c)
        x-=1
        y-=1
        w = min(w+2,arr.shape[1]-x)
        h = min(h+2,arr.shape[0]-y)
        if x<0: 
            x=0
        if y<0: 
            y=0

        #print(y,x,w,h)
        trim = arr[y:y+h,x:x+w]
        newim = np.zeros((h,w),dtype=np.uint8)
        mask = np.zeros((h,w),dtype=np.uint8)+1
        cv2.fillPoly(mask,pts = [c] + insides, color=0, offset=(-x,-y), lineType=4)
        #print(trim,y,x,w,h)
        #try:
        newim = (255 - (trim == mask)*255).astype(np.uint8)
        #except:
            #save(arr,'output.png')
            #sys.exit(1)
        offset = [x,y]
        iso.append((newim,offset))

    return iso


def separate_rectangle_out(arr):
    #squ = arr['contour'][:]
    squ = get_outer_rect(arr['img'],arr['contour'])
    #grow_rect_by_one(squ)

    outside = np.copy(arr['img'])

    cv2.fillPoly(outside, [squ], 255)

    outside = wrap_image(outside,arr)
    return outside

def separate_rectangle_in(arr):
    squ = get_inner_rect(arr['img'],arr['contour'])

    outside = np.zeros(arr['img'].shape).astype(arr['img'].dtype)
    inside = np.zeros(arr['img'].shape).astype(arr['img'].dtype)

    cv2.fillPoly(inside, [squ], 1)
    outside = (inside != 1) * 255
    inside = ((inside == 1) * 255).astype(arr['img'].dtype)

    result = cv2.bitwise_and(arr['img'], inside)

    outside = wrap_image((result + outside).astype(arr['img'].dtype),arr)
    return outside


def separate_rectangles(inp):
    outsides = []
    for x in inp:
        outsides.append(separate_rectangle_out(x))
        outsides.append(separate_rectangle_in(x))
    return outsides

def rotate_right(inp):
    for x in inp:
        x['img'] = cv2.rotate(x['img'], cv2.ROTATE_90_CLOCKWISE)
        x['rotated'] = True

def rotate_left(inp):
    for x in inp:
        x['img'] = cv2.rotate(x['img'], cv2.ROTATE_90_COUNTERCLOCKWISE)
        x['rotated'] = True


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



