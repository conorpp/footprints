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
    print('extracting..')
    coms = extract_components(arrs)
    #trim_images(coms)

    #cut_linking_lines(coms)
    #coms = extract_components(coms)
    #trim_images(coms2)
    #while len(coms2) > len(coms):
        #print('coms: %d, coms2: %d'%(len(coms),len(coms2)))
        #coms = coms2
        #cut_linking_lines(coms)
        #coms2 = extract_components(coms)
        #trim_images(coms2)

    #print('coms2: %d'%(len(coms2)))
    return coms


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
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# requires contours to all be pixel-to-pixel
def get_partial_lines_from_contour(contour):
    def segment(c,i):
        return np.reshape(c[i:i+2],(2,2))

    c = contour
    cx,cy = centroid(contour)

    #cx *= 2
    #cy *= 2
    print('ORIGIN:',cx,cy)
    traces = []
    for i in range(0,len(c)-1):
        traces.append(segment(c,i))
    traces.append(np.array([c[-1][0],c[0][0]]))

    #traces = traces[:len(traces)]+traces[len(traces):]
    traces = traces[int(len(traces)/2):]+traces[:int(len(traces)/2)]
    #traces.reverse()

    lastp1,lastp2 = traces[-1]
    lastslop = 1
    route = []

    # calculate distance from origin for each contour point
    for i,(p1,p2) in enumerate(traces):

        ni = p2[0] - lastp2[0]
        nj = p2[1] - lastp2[1]
        nk = 0

        li = p2[0] - lastp2[0]
        lj = p2[1] - lastp2[1]
        lk = 0

        dist1 = line_len((lastp1,lastp2))
        dist2 = line_len((p1,p2))
        ang = angle_between((li,lj,0),(ni,nj,0))
        route.append([dist1,dist2,ang,(lastp1,lastp2),(p1,p2),line_len((p1,(cx,cy)))])

        lastp1 = p1
        lastp2 = p2

    origin_dists = np.zeros(len(route))
    for i in range(0,len(route)):
        origin_dists[i] = route[i][5]

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
            if count/len(diffs) > .45:
                # trim the edges
                while diffs[-1] != mode:
                    diffs.pop()
                    i -= 1
                line = traces[center-i+1:center+i]
                line[-1] = np.copy(line[-1])
                line[-1][1] = line[0][0]
                good_traces += line
                #print('good diff len', len(diffs), 'mode %',count/len(diffs))
            #else:
                #print('bad diff len', len(diffs), 'mode %',count/len(diffs))

    #print(traces)

    #maxima = np.zeros(origin_dists.shape)
    #maxima[minlocs] = -1
    #maxima[maxlocs] = 1

    #plt.plot(origin_dists/20 -5)
    #plt.plot(maxima)
    #plt.show()
    #sys.exit(0)
    return good_traces

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



def navigate_lines(lines):
    
    def segment(c,i):
        return np.reshape(t['ocontour'][i:i+2],(2,2))

    for x in lines:
        print(x['img'].shape)
        #x['target'] = False
        x['target'] = False
        #x['traces'] = get_partial_lines_from_contour(x['ocontour'])
    lines[1]['target'] = True
    lines[1]['traces'] = get_partial_lines_from_contour(lines[1]['ocontour'])

