import sys,os,json,argparse
from scipy.signal import argrelextrema
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
            if new is not None:
                new = wrap_image(new,x)
                out.append(new)
            old = wrap_image(old,x)
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
            #print('faulty image',faulty_id)
            newim = None
            #save(im,'out/fault%d.png'%faulty_id)


    if dim == 0:
        im = np.transpose(im)
        if newim is not None: newim = np.transpose(newim)

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
    coms = extract_components(arrs)
    return coms


def extract_components(arrs):
    submaps = []
    for arr in arrs:
        img = arr['img']
        #track_map = np.zeros(img.shape[:2],dtype=np.uint8)
        _submaps = get_isolated_images(img)
        submaps += [wrap_image(x[0],arr,x[1]) for x in _submaps]

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

        if i == 85:
            print('insides',len(insides))
        #print(y,x,w,h)
        trim = arr[y:y+h,x:x+w]
        newim = np.zeros((h,w),dtype=np.uint8)
        mask = np.zeros((h,w),dtype=np.uint8)+1
        cv2.fillPoly(mask,pts = [c] + insides, color=0, offset=(-x,-y), lineType=8)
        #print(trim,y,x,w,h)
        #try:
        newim = (255 - (trim == mask)*255).astype(np.uint8)
        #except:
            #save(arr,'output.png')
            #sys.exit(1)
        offset = [x,y]
        iso.append((newim,offset,c))

    return iso


def separate_rectangle_out(arr):
    #squ = arr['contour'][:]
    squ = get_outer_rect(arr['img'],arr['contour'])
    #grow_rect_by_one(squ)

    outside = np.copy(arr['img'])

    outside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255

    outside = wrap_image(outside,arr)
    return outside

def separate_rectangle_in(arr):
    squ = get_inner_rect(arr['img'],arr['contour'])

    inside = np.zeros(arr['img'].shape, dtype=np.uint8)

    inside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255

    result = cv2.bitwise_and(arr['img'], inside)

    outside = wrap_image((result + (inside != 255) * 255).astype(np.uint8),arr)
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
    cut = []
    notcut = []
    for arr in arrs:
        img = arr['img']
        if cut_linking_line(img) > 0:
            cut.append(arr)
        else:
            notcut.append(arr)
    return cut,notcut

def cut_linking_line(arr):
    #TODO derive these from something
    perpendicular_line_length = 8
    min_line_length = 10
    variation = 3
    cuts = 0

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
            cuts += 1


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
            cuts += 1
    return cuts

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180.0/math.pi

def smooth(data):
    return smooth_avg(data,5)
def smooth_avg(y, window):
    box = np.ones(window)/window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# requires contours to all be pixel-to-pixel
def get_partial_lines_from_contour(contour, center=None, headstart=True):
    def segment(c,i):
        return np.reshape(c[i:i+2],(2,2))


    c = contour
    if center is None:
        cx,cy = centroid(contour)
    else:
        cx,cy = center

    #print('ORIGIN:',cx,cy)
    traces = []
    for i in range(0,len(c)-1):
        traces.append(segment(c,i))
    traces.append(np.array([c[-1][0],c[0][0]]))

    if headstart:
        traces = traces[:len(traces)]+traces[len(traces):]
    else:
        # TODO do this better
        traces = traces[int(len(traces)/2):]+traces[:int(len(traces)/2)]
    #traces.reverse()

    lastp1,lastp2 = traces[-1]
    lastslop = 1
    route = []

    # calculate distance from origin for each contour point
    for i,(p1,p2) in enumerate(traces):

        #ni = p2[0] - lastp2[0]
        #nj = p2[1] - lastp2[1]
        #nk = 0

        #li = p2[0] - lastp2[0]
        #lj = p2[1] - lastp2[1]
        #lk = 0

        #dist1 = line_len((lastp1,lastp2))
        #dist2 = line_len((p1,p2))
        #ang = angle_between((li,lj,0),(ni,nj,0))
        #route.append([dist1,dist2,ang,(lastp1,lastp2),(p1,p2),line_len((p1,(cx,cy)))])
        route.append(line_len((p1,(cx,cy))))

    origin_dists = np.zeros(len(route))
    for i in range(0,len(route)):
        origin_dists[i] = route[i]

    origin_dists = smooth(origin_dists)

    maxlocs = argrelextrema(origin_dists, np.greater)
    minlocs = argrelextrema(origin_dists, np.less)
    locs = sorted(maxlocs[0].tolist() + minlocs[0].tolist())
    # get locations for potential lines based on maxima
    pot_lines = []
    for i in range(1,len(locs)-1):
        start = locs[i-1]
        center = locs[i]
        end = locs[i+1]
        pot_lines.append([center,min(center-start,end-center)]) # center, scanning distance

    # filter potential lines based on "flat/thin-ness" at the start
    good_traces = []
    good_traces_start_points = []
    for center,dist in pot_lines:
        diffs = []
        for i in range(1, dist+1):
            pl = traces[center-i][0]
            pr = traces[center+i][0]
            diff = line_len((pl,pr))
            if diff > 6.5: break  # TODO normalize this
            diffs.append(diff)

        if len(diffs) > 9: # TODO normalize this
            mode,count = stats.mode(diffs)
            # only take items with majority taken by mode
            if count/len(diffs) > .45:
                # trim the edges
                while diffs[-1] != mode:
                    diffs.pop()
                    i -= 1
                line = traces[center-i+1:center+i]
                #line = line[:i-3] + line[i+3:]
                #line = line[:len(line)-3]
                #line[-1] = np.copy(line[-1])
                #line[-1][1] = line[0][0]
                good_traces.append(line)
                good_traces_start_points.append(traces[center][0])
                #print('good diff len', len(diffs), 'mode %',count/len(diffs))
            #else:
                #print('bad diff len', len(diffs), 'mode %',count/len(diffs))

    #print(traces)

    maxima = np.zeros(origin_dists.shape)
    maxima[minlocs] = -1
    maxima[maxlocs] = 1

    #plt.plot(origin_dists/20 -5)
    #plt.plot(maxima)
    #plt.show()
    #sys.exit(0)
    return (good_traces,good_traces_start_points)
    #return (good_traces)

    # this method detects straight horizontal/vertical lines well
    #route_prepend = []
    #if route[0][0] < 10:
        #ang_accum = 0
        #for i in range(len(route)-1,-1,-1):
            #r = route[i]
            #route_prepend = [r] + route_prepend
            #if r[0] >= 10:
                #break
            #if abs(ang_accum) >= 180:
                #break
            #ang_accum += r[2]

    #route = route_prepend + route

    #lastdist = 0.0
    #for offset in range(0,len(route)):
        ##offset = 7
        #dist1 = route[offset][0]
        #ang_accum = 0.0
        #ang_dist = 0.0
        #hist = []
        #print('dist1: %.2f:' % (dist1,))
        #for _,dist2,ang,l1,l2,m1,m2 in route[offset:]:
            #ang_accum += ang
            #hist.append(l1)
            #hist.append(l2)
            #print('+dist: %.2f, +ang: %.2f (%.2f)' % (dist2,ang,ang_accum))

            #if abs(int(ang_accum)) == 180 or abs(int(ang_accum)) == 0:
                #if min(dist1,dist2) > 5 and ang_dist < 25:  # min line length
                    #traces += hist
                    #print('DETECTED LINE')
                #break
            #elif abs(ang_accum) > 180:
                #break
            #ang_dist += dist2

        ##traces += hist
        #print('dist1: %.2f, dist2: %.2f, ang_dist %.2f' % (dist1,dist2,ang_dist))
        ##break


def find_line_features(lines):
    def neighboring_black_pixel(im,pt):
        x,y = pt
        if im[y-1,x] == 0:
            return (x,y-1)
        if im[y+1,x] == 0:
            return (x,y+1)
        if im[y,x-1] == 0:
            return (x-1,y)
        if im[y,x+1] == 0:
            return (x+1,y)
        if im[y-1,x-1] == 0:
            return (x-1,y-1)
        if im[y+1,x-1] == 0:
            return (x-1,y+1)
        if im[y-1,x+1] == 0:
            return (x+1,y-1)
        if im[y+1,x+1] == 0:
            return (x+1,y+1)
        raise RuntimeError('No neighboring black pixel')

    newims = []
    oldims = []
    for l in lines:
        #l['target'] = False
        l['target'] = True
        scantype = l['line-scan-attempt']
        
        if scantype == 1:
            oldims.append(l)
            continue

        scan_points = [
                None,
                (0,0),
                (-100,100),
                (0,l['img'].shape[0]),
                (l['img'].shape[1],0),
                (l['img'].shape[1],l['img'].shape[0]),
                ]

        for i in range(0,5):
            traces, startpoints = get_partial_lines_from_contour(l['ocontour'], scan_points[i])
            if len(traces): break

        if len(traces) == 0:
            for i in range(0,5):
                traces, startpoints = get_partial_lines_from_contour(l['ocontour'], scan_points[i],False)
                if len(traces): break

            if len(traces) == 0:
                l['line-scan-attempt'] = 1
                oldims.append(l)
                continue

        l['line-scan-attempt'] = scantype + 1

        l['traces'] = (traces,startpoints)

        ests = []
        for i in range(0, len(traces)):
            tset = traces[i]
            pts = np.array([t[0] for t in tset])
            startp = startpoints[i]

            blackp = neighboring_black_pixel(l['img'], startp)

            # for horizontal or vertical lines
            hv_line,vert = grow_line(l['img'],None,blackp)

            if line_len(hv_line) > 9:
                if line_len((hv_line[0], startp)) > line_len((hv_line[1], startp)):
                    p1 = hv_line[1]
                    p2 = hv_line[0]
                else:
                    p1 = hv_line[1]
                    p2 = hv_line[0]
                m = 0
                if vert:
                    m = (p2[1] - p1[1])*1000

                b = p1[1] - m*p1[0]

                line_est = [(p1,p2),m,b,False]
            else:
                # for other lines
                line_est = estimate_line(pts,startp)

            ests.append(line_est)

        oldim = np.copy(l['img'])

        # Sort from max to least so that larger features will include small ones
        ests = sorted(ests, key = lambda x: line_len(x[0]), reverse = True)
        l['line-estimates'] = ests
        features = []
        for e in ests:
            new_features = get_pixels_following_line(oldim,e)
            #print(len(new_features),'features')
            features += new_features
            if len(new_features):
                newim = np.zeros(oldim.shape,dtype=np.uint8) + 255
                for feat in new_features:
                    for x,y in feat:
                        # remove from old image
                        oldim[y,x] = 255
                        # add to extracted image
                        newim[y,x] = 0
                newims.append(wrap_image(newim,l))

        if len(features):
            oldims.append(wrap_image(oldim,l))
        else:
            oldims.append(l)

        l['features'] = features

    return newims,oldims
#def extract_line_features(line):
    #oldim = 
    #for feature in line['features']:
        #newim = np.zeros()
def assign_best_fit_lines(lines):
    for x in lines:
        pts = np.argwhere(x['img'] == 0)
        #print(pts)
        l = estimate_line(pts)[0]
        x['line'] = np.array(((l[0][1],l[0][0],),(l[1][1],l[1][0],)))


# line of best fit to set of contour points
def estimate_line(pts, startp=None):
    line = cv2.fitLine(pts,cv2.DIST_L12,0,.01,.01)
    xlim1,ylim1,xlim2,ylim2 = cv2.boundingRect(pts)
    xlim2+=xlim1
    ylim2+=ylim1
    vx,vy,x0,y0 = line
    m = vy[0]/vx[0]
    b = y0[0] - m*x0[0]

    y2 = m * xlim2 + b
    y1 = m * xlim1 + b
    if abs(y2-y1) < (ylim2 - ylim1):
        p1 = (int(xlim1), int(y1))
        p2 = (int(xlim2), int(y2))
    else:
        x1 = (ylim1 - b)/m
        x2 = (ylim2 - b)/m
        p1 = (int(x1), int(ylim1))
        p2 = (int(x2), int(ylim2))

    if startp is None:
        return [(p1,p2),m,b,False]
    
    # first point shall be "start of line"
    if line_len((startp,p1)) > line_len((startp,p2)):
        #ests.append([(p2,p1),m,b,False])
        return [(p2,p1),m,b,False]
    else:
        return [(p1,p2),m,b,False]

def infinity_line(l):
    p1,p2 = l
    if (p2[0]-p1[0]) == 0:
        m = 100
    else:
        m = (p2[1]-p1[1])/(p2[0]-p1[0])

    b = p1[1] - m*p1[0]

    p1 = (1000 * -m, int(m * 1000 * -m + b))
    p2 = (1000 * m, int(m * 1000 * m + b))

    return np.array((p1,p2))


def get_pixels_following_line(img,est):
    def addpixlocs(img, tmplocs, it, dim, inc):
        #try:
            dimp = (dim + 1) & 1
            k = 1 if inc > 0 else 0

            if it[dim]+k == img.shape[dimp]:
                k = 0

            while img[it[1] + (inc*dim)*k, it[0] + (inc*dimp)*k] == 0:
                tmplocs.append(np.array((it[0] + (inc*dimp)*k, it[1] + (inc*dim)*k)))
                k += 1
                #if k > 3: 
                    #del tmplocs[:]
                    #return k
            return k
        #except Exception as e:
            #print(e)
            #save(img,'output.png')
            #sys.exit(1)

    alllocs = []
    start = est[0][0]
    end = est[0][1]
    m = est[1]
    b = est[2]

    goup = (start[1] > end[1])
    goright = (start[0] < end[0])
    isvert = (abs(m)>1)

    it = np.copy(start)
    lastlen = 0
    count = 0
    gapdist = 0
    linelocs = []
    ksums = []
    lastcount = 0

    locs   = []
    counts = []

    for j in range(0,int(line_len(((0,0),img.shape)))):
        if it[1] < 0 or it[1] >= img.shape[0]:
            break
        if it[0] < 0 or it[0] >= img.shape[1]:
            break

        # go left
        leftlocs = []
        rightlocs = []
        dim = 0 if isvert else 1
        k1 = addpixlocs(img,leftlocs,it,dim,-1)
        if k1 < 4:
            k2 = addpixlocs(img,rightlocs,it,dim,1)
        else:
            k2 = 4

        locs.append((leftlocs,rightlocs))
        counts.append((k1,k2))

        # vert
        if isvert:
            if goup:
                it[1] -= 1
            else:
                it[1] += 1

            it[0] = int(((it[1] - b) / m))

        else:

            if goright:
                it[0] += 1
            else:
                it[0] -= 1
            it[1] = int((it[0] * m + b))

    start_skipping = False
    length = 0
    ksums = []
    for i in range(0,len(locs)):
        lefts,rights = locs[i]
        lc,rc = counts[i]
        if lc < 4 and rc < 4 and lc and rc:
            #if start_skipping:
                #continue
            ##check starting bound
            #if len(linelocs) == 0:
                #if i > 3:
                    #past = [l+r for l,r in counts[i-3:i]]
                    #if 0 not in past:
                        #if max(past) < 20:
                            #start_skipping = True
            linelocs += lefts
            linelocs += rights
            length += 1
            #ksums.append(lc+rc)
        else:
            start_skipping = False
            if length > 3:
                #past = [l+r for l,r in counts[i-3:i]]
                #if 0 not in past:
                    #if max(past) < 0:
                        #continue
                #metric = len(set(ksums))/float(len(ksums))
                alllocs.append(linelocs)

            #ksums = []
            linelocs = []
            length = 0

    return alllocs

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)
    arr = load_image(sys.argv[1])
    if len(arr.shape) > 2:
        arr = polarize(arr)
 
    print(arr.shape)

    wtfwhy = np.array(np.copy(arr).tolist(),dtype=np.uint8)
    mat, contours, hier = cv2.findContours(wtfwhy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ims = get_isolated_images(arr)


    orig = np.zeros(arr.shape, dtype=np.uint8)+255
    for i,x in enumerate(ims):
        oldim = x[0]
        oj,oi = x[1]
        for i in range(0,oldim.shape[0]):
            for j in range(0,oldim.shape[1]):
                v = oldim[i,j]
                orig[i+oi,j+oj] = v

        save(oldim,'out/im%d.png' % (i))

    orig = color(orig)


    print(len(contours),'contours')
    print(len(ims),'islands')
    for i,x in enumerate(ims):
        x = x[2]
        if i in range(85,85+1):
            cv2.drawContours(orig,[x],0,[255,0,0],1)
        #if i in range(81,82):
            #cv2.drawContours(orig,[x],0,[255,0,0],1)

    save(orig,'output.png')

