import sys,os,json,argparse,time,math

from scipy import signal,stats
import numpy as np
import cv2

import analyzers
from utils import *

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def line_overlaps(arr,line,ref):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    s = np.sum(arr[y1:y2+1,x1:x2+1])

    if x1 == x2:
        l1 = abs(y2-y1)
        l2 = abs(ref[1][1]-ref[0][1])
    else:
        l1 = abs(x2-x1)
        l2 = abs(ref[1][0]-ref[0][0])

    if l1 > l2:
        s -= (l1-l2)

    return s == 0

def line_trace(arr,line,padding,dim):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    if y1==y2:
        y1 -= padding
        if y1 < 0: y1 = 0
        y2 += padding
        s = np.sum(arr[y1:y2+1,x1:x2+1],axis=dim)
    if x1==x2:
        x1 -= padding
        if x1 < 0: x1 = 0
        x2 += padding
        s = np.sum(arr[y1:y2+1,x1:x2+1],axis=dim)


    return s

def rect_area(r):
    #r = [tl[1],tr[1],br[1],bl[1],tl[1]]
    dx = abs(r[0][0] - r[2][0])
    dy = abs(r[0][1] - r[2][1])
    return dx * dy


def get_intersects(im,margin1,margin2):
    y1 = scan_dim(im,0)
    y1mean = np.mean(y1)
    y1f = butter_highpass_filter(y1,1,25)
    y2 = scan_dim(im,1)
    y2mean = np.mean(y2)
    y2f = butter_highpass_filter(y2,1,25)

    #print('dim is ',len(y1),'long')
    #plt.subplot(211,title='columns collapsed')
    #plt.plot(y1)
    #plt.plot([y1mean]*len(y1))
    #plt.subplot(212,title='rows collapsed') # creates 2nd subplot with yellow background
    #plt.plot(y2)
    #plt.plot([y2mean]*len(y2))
    #save(im,'output.png')
    #plt.show()
    #sys.exit(0)

    y1f = y1f > margin1
    y2f = y2f > margin2
    xlocs = np.where(y1f)[0]
    xlocs = np.split(xlocs, np.where(np.diff(xlocs) != 1)[0]+1)
    tmp = []
    for i,v in enumerate(xlocs):
        if len(v):
            m = 0
            mi = v[0]
            for x in v:
                if y1[x] > m:
                    m = y1[x]
                    mi = x
            tmp.append(mi)
    xlocs = tmp

    ylocs = np.where(y2f)[0]
    ylocs = np.split(ylocs, np.where(np.diff(ylocs) != 1)[0]+1)
    tmp = []
    for i,v in enumerate(ylocs):
        if len(v):
            m = 0
            mi = v[0]
            for x in v:
                if y2[x] > m:
                    m = y2[x]
                    mi = x
            tmp.append(mi)
    ylocs = tmp

    return xlocs,ylocs



def get_line_signals(im):

    return y1,y2

def get_corner_map(arr, xlocs,ylocs):
    corner_map = np.zeros((len(xlocs),len(ylocs)))
    for i,x in enumerate(xlocs):
        for j,y in enumerate(ylocs):
            if arr[y,x] == 0:
                corner_map[i,j] = 1
    return corner_map


def get_line_map(arr,corner_map,xlocs,ylocs):
    line_map_h = np.zeros((max(0,len(xlocs)-1),len(ylocs)))    # dual of corner map
    line_map_v  = np.zeros((len(xlocs),max(0,len(ylocs)-1)))    # dual of corner map


    # neighboring black pixel intersects
    for y in range(0,len(ylocs)):
        for x in range(0,len(xlocs)-1):
            if corner_map[x,y] and corner_map[x+1,y]:
                l = [(xlocs[x],ylocs[y]),(xlocs[x+1],ylocs[y])]

                if line_exists(arr,l):
                    #lines.append(l)
                    line_map_h[x,y] = 1

    for y in range(0,len(ylocs)-1):
        for x in range(0,len(xlocs)):
            if corner_map[x,y] and corner_map[x,y+1]:
                l = [(xlocs[x],ylocs[y]),(xlocs[x],ylocs[y+1])]

                if line_exists(arr,l):
                    #lines.append(l)
                    line_map_v[x,y] = 1
    return line_map_h,line_map_v



def get_corners(line_map_h,line_map_v,xlocs,ylocs):

    tlcorners = []
    trcorners = []
    blcorners = []
    brcorners = []
    # detect the corners
    for y in range(0,len(ylocs)-1):
        for x in range(0,len(xlocs)-1):
            if line_map_h[x,y]:
                # top left corners
                if line_map_v[x,y]:
                    p1 = (xlocs[x],ylocs[y+1])
                    p2 = (xlocs[x],ylocs[y])
                    p3 = (xlocs[x+1],ylocs[y])
                    tlcorners.append([p1,p2,p3])

                # top right corners
                if line_map_v[x+1,y]:
                    p1 = (xlocs[x],ylocs[y])
                    p2 = (xlocs[x+1],ylocs[y])
                    p3 = (xlocs[x+1],ylocs[y+1])
                    trcorners.append([p1,p2,p3])

    for y in range(len(ylocs)-2,-1,-1):
        for x in range(len(xlocs)-2,-1,-1):
            if line_map_h[x,y+1]:
                # bot right corners
                if line_map_v[x+1,y]:
                    p1 = (xlocs[x+1],ylocs[y])
                    p2 = (xlocs[x+1],ylocs[y+1])
                    p3 = (xlocs[x],ylocs[y+1])
                    brcorners.append([p1,p2,p3])

                # bot left corners
                if line_map_v[x,y]:
                    p1 = (xlocs[x+1],ylocs[y+1])
                    p2 = (xlocs[x],ylocs[y+1])
                    p3 = (xlocs[x],ylocs[y])
                    blcorners.append([p1,p2,p3])

    return tlcorners, trcorners, brcorners, blcorners


def merge_corners(arr,tlcorners, trcorners, brcorners, blcorners):
    rectangles = []
    for tl in tlcorners:
        for tr in trcorners:
            if tl[1][1] != tr[1][1] or tl[1][0] == tr[1][0]:
                continue

            if not line_exists(arr,([tl[1],tr[1]]) ):
                continue

            for br in brcorners:
                if br[1][0] != tr[1][0] or br[1][1] == tr[1][1]:
                    continue

                if not line_exists(arr,([br[1],tr[1]]) ):
                    continue


                for bl in blcorners:
                    if br[1][1] != bl[1][1] or br[1][0] == bl[1][0]:
                        continue

                    if tl[1][0] != bl[1][0] or tl[1][1] == bl[1][1]:

                        continue

                    if not line_exists(arr,([br[1],bl[1]]) ):
                        continue
                    if not line_exists(arr,([tl[1],bl[1]]) ):
                        continue

                    r = [tl[1],tr[1],br[1],bl[1],tl[1]]
                    rectangles.append(r)
    return rectangles


def detect_overlap(arr,rectangles,ref):
    skips = 0
    rect_map = np.zeros(arr.shape)+1
    rects = []
    overlaps = []

    #print('ref')
    #print(ref)

    for x in rectangles:
        count = 0
        #print('x')
        #print(x)

        for i in range(0,4):
            side_ref = ref[i:2+i]
            side = x[(i + 2) % 4:(i + 2 ) % 4 + 2]
            #print((i,i+2), 'vs', ((i+2)%4, (i+2)%4 +2))
            #print(side_ref[0],side_ref[1],'vs',side[0],side[1],)
            if (side == side_ref).all():
                np_rm(rectangles,x)
                return x

            elif (side == np.flip(side_ref,0)).all():
                np_rm(rectangles,x)
                return x
     
    return None

def np_rm(l,item):
    for i,v in enumerate(l):
        if item is v:
            del l[i]
            return

# must be sorted from least area to most area prior to this
def remove_super_rects(arr,rectangles):
    good = []
    rect_map = np.zeros(arr.shape)
    
    while len(rectangles):
        rect_map[:,:] = 1
        ref = rectangles.pop(0)
        good.append(ref)
        for i in range(0,4):
            side = ref[0+i:2+i]
            set_line(rect_map,side)

        for x in rectangles:
            count = 0
            for i in range(0,4):
                side = x[0+i:2+i]
                side_ref = ref[0+i:2+i]
                if line_overlaps(rect_map,side,side_ref):
                    count += 1
                    if count > 1:
                        np_rm(rectangles,x)
                        break
    return good

#order shouldnt matter
def remove_side_rects(arr,rectangles):
    good = []
    bad = []
    rect_map = np.zeros(arr.shape)
    
    while len(rectangles):
        rect_map[:,:] = 1
        ref = rectangles.pop(0)
        add = True

        for i in range(0,4):
            side = ref[0+i:2+i]
            set_line(rect_map,side)

        for x in rectangles:
            if not add: break
            count = 0
            for i in range(0,4):
                side = x[0+i:2+i]
                side_ref = ref[0+i:2+i]
                if line_overlaps(rect_map,side,side_ref):
                    if line_len(side_ref) > line_len(side):
                        np_rm(rectangles,x)
                        bad.append(x)
                    else:
                        add = False
                    break
        if add:
            good.append(ref)
        else:
            bad.append(ref)

    return good,bad

def take_match(rectangles, overlap):

    r = overlap
    rect = r[0]
    i = r[1]
    side = rect[i[0]:i[1]]
    for r2 in rectangles:
        if (side == r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2]).all():
            np_rm(rectangles,r2)
            return (rect,r2)
        elif (side == np.flip(r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2],0)).all():
            np_rm(rectangles,r2)
            return (rect,r2)


def reconcile_overlaps(arr,rectangles,overlap_pair):
    rejects = []
    inv = (arr==0)
    r1means = np.zeros(4)
    r2means = np.zeros(4)
    r1vars = np.zeros(4)
    r2vars = np.zeros(4)
    r1,r2 = overlap_pair
    r1singlemean = None
    r2singlemean = None
    for ii in range(1,10):
        r1t = [line_trace(inv,r1[x:x+2],ii,x&1) for x in range(0,4)]
        r2t = [line_trace(inv,r2[x:x+2],ii,x&1) for x in range(0,4)]

        for i,x in enumerate(r1t):
            r1means[i] = np.mean(x)
            r1vars[i] = np.var(x)
        for i,x in enumerate(r2t):
            r2means[i] = np.mean(x)
            r2vars[i] = np.var(x)

        r1svar = np.var(r1means)
        r2svar = np.var(r2means)

        r1smean = np.mean(r1means)
        r2smean = np.mean(r2means)

        #print('means',r1smean,r2smean)
        #print('vars',r1svar,r2svar)
        rej = None

        if r1svar == 0 or r2svar == 0:
            if r1svar > .1:
                rej = r1
            if r2svar > .1:
                rej = r2
        if abs(r1smean - r2smean) >= 1.5:
            if r1smean < r2smean:
                rej = r1
            else:
                rej = r2
        #print('i',ii)
        if rej is not None:
            #print('REJECTED-----------')
            rejects.append(rej)
            if rej is r1:
                #print(' mean between sides: %.2f' % (r1smean))
                #print(' var between sides: %.2f' % (r1svar))
                rectangles.append(r2)
            else:
                #print(' mean between sides: %.2f' % (r2smean))
                #print(' var between sides: %.2f' % (r2svar))

                rectangles.append(r1)
            break

        if ii == 1:
            r1singlemean = r1smean
            r2singlemean = r2smean
        else:
            #print('r1single', r1singlemean * ii * .7, )
            #print('r2single', r2singlemean * ii * .7)
            if (r1singlemean * ii * .7) > r1smean and (r2singlemean * ii * .7) > r2smean:
                #print('stopping early')
                break



    if rej is None:
        verts = np.concatenate((r1,r2))
        vx = verts[:,0]
        vy = verts[:,1]

        vmaxxloc = np.where(vx == np.max(vx))
        vminxloc = np.where(vx == np.min(vx))

        vmaxx = verts[vmaxxloc]
        vminx = verts[vminxloc]

        vminx_y = vminx[:,1]
        bl = vminx[np.where(vminx_y == np.max(vminx_y))][0]
        tl = vminx[np.where(vminx_y == np.min(vminx_y))][0]
        
        vmaxx_y = vmaxx[:,1]
        br = vmaxx[np.where(vmaxx_y == np.max(vmaxx_y))][0]
        tr = vmaxx[np.where(vmaxx_y == np.min(vmaxx_y))][0]

        #print('merged rect: ', tl, tr, br, bl)
        rectangles.append(np.array([tl,tr,br,bl,tl]))
 
    return rejects

# group together rectangles of close-to-same dimensions
def group_rects(rects,margin=12):
    # [ [xdim,ydim,[rects,...] ],... ]
    groups = []

    for r in rects:
        xdim = abs(r[0][0] - r[2][0])
        ydim = abs(r[0][1] - r[2][1])
        added = False
        for dx,dy,gro in groups:
            if xdim < (dx+margin) and xdim > (dx-margin):
                if ydim < (dy+margin) and ydim > (dy-margin):
                    gro.append(r)
                    added = True
                    break
        if not added:
            groups.append([xdim,ydim,[r]])

    return groups



def block_small_rects(arr,rectangles):
    good = []
    a2 = arr.shape[0] * arr.shape[1]
    rects = [x for x in rectangles if rect_area(x)/a2 > .0002]
    # block rects with few pixel dim or less
    for x in rects:
        xdim = abs(x[0][0] - x[2][0])
        ydim = abs(x[0][1] - x[2][1])
        if xdim > 5 and ydim > 5:
            good.append(x)
    return good

# must call analyzers.init first
def get_rectangles(arr):
    xlocs, ylocs = get_intersects(arr, (arr.shape[0]*.03), (arr.shape[1]*.03))

    corner_map = get_corner_map(arr,xlocs,ylocs)

    # dual of corner map
    line_map_h, line_map_v = get_line_map(arr,corner_map,xlocs,ylocs)

    # intersects over black pixel
    tlcorners, trcorners, brcorners, blcorners = get_corners(line_map_h,line_map_v,xlocs,ylocs)

    # detect the rectangles
    rectangles = merge_corners(arr,tlcorners, trcorners, brcorners, blcorners)

    rectangles = [np.array(x) for x in rectangles]

    return rectangles


def coalesce_rectangles(arr, rectangles):
    #rejects = []
    last_len = -1

    rectangles = block_small_rects(arr,rectangles)

    rectangles = sorted(rectangles,key = lambda x: rect_area(x))

    rectangles = remove_super_rects(arr,rectangles)

    islands = []
    while len(rectangles):
        r = rectangles.pop(0)
        adj_r = detect_overlap(arr,rectangles,r)

        if adj_r is None:
            islands.append(r)
            continue

        #rejects += reconcile_overlaps(arr,rectangles,(r,adj_r))
        reconcile_overlaps(arr,rectangles,(r,adj_r))

    rectangles = islands

    rectangles, bad = remove_side_rects(arr,rectangles)
    #rejects += bad

    return rectangles

def separate_grouped_rectangles(arr, rectangles, *args):

    line_thickness = analyzers.PARAMS['line-thickness']

    groups = group_rects(rectangles, line_thickness * 6)
    rectangles = []

    for xdim,ydim,gro in groups:
        rectangles += gro
        # cutout rectangle groups
        if len(gro) > 1:
            for x in gro:
                outer = analyzers.get_outer_rect(arr,x)
                inner = analyzers.get_inner_rect(arr,x)
                cv2.drawContours(arr,[outer],0,255,1)
                cv2.drawContours(arr,[inner],0,255,1)
                for y in args:
                    cv2.drawContours(y,[outer],0,[255,255,255],1)
                    cv2.drawContours(y,[inner],0,[255,255,255],1)

def separate_largest_rectangle(arr, rectangles, *args):
    if not len(rectangles):
        return
    r = max(rectangles, key = lambda x : rect_area(x))
    #print(r)
    outer = analyzers.get_outer_rect(arr,r)
    inner = analyzers.get_inner_rect(arr,r)
    cv2.drawContours(arr,[outer],0,255,1)
    cv2.drawContours(arr,[inner],0,255,1)
    for y in args:
        cv2.drawContours(y,[outer],0,[255,255,255],1)
        cv2.drawContours(y,[inner],0,[255,255,255],1)



def preprocess(arr,*args):
    arr[:,:2]=255
    arr[:,-2:]=255
    arr[:2,:]=255
    arr[-2:,:]=255

    rectangles = get_rectangles(arr)

    rectangles = coalesce_rectangles(arr, rectangles)

    rectangles = [convert_rect_contour(x) for x in rectangles]

    separate_grouped_rectangles(arr, rectangles, *args)
    separate_largest_rectangle(arr, rectangles, *args)
    return arr

if __name__ == '__main__':
    """ make results for blog post """
    import plotly
    from plotly import tools
    from plotly.graph_objs import Scatter,Layout
    from analyzers import analyze_rectangles,PARAMS
    import image_handling

    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)
    arr,orig = image_handling.init(sys.argv[1])
    print(arr.img.shape)
    analyze_rectangles([arr])
    im = arr.img
    orig = np.copy(PARAMS['orig'])

    y1 = scan_dim(im,0)
    y1 = np.sum(im == 0,axis=0)     # row sum
    y1mean = np.mean(y1)
    y1f = butter_highpass_filter(y1,1,25)
    y2 = np.sum(im == 0,axis=1)     # column sum
    y2mean = np.mean(y2)
    y2f = butter_highpass_filter(y2,1,25)

    trace = Scatter(
        y = y1,
        mode = 'lines'
    )

    trace2 = Scatter(
        y = y1f,
        mode = 'lines'
    )

    xlocs, ylocs = get_intersects(im, (im.shape[0]*.03), (im.shape[1]*.03))
    show(~im)
    print(xlocs)
    print(ylocs)

    for x in xlocs:
        orig[:,x] = (255,0,0)

    #for y in ylocs:
        #orig[y,:] = (255,0,0)



    if 1:
        fig = [trace,trace2]

        plotly.offline.plot(fig, auto_open=True, filename='blog.html')
        input_img = im
        orig_img = orig

        grey_img = cv2.GaussianBlur(input_img,(5, 5),0)
        edges = cv2.Canny(grey_img, 50, 150)

        rho = 3
        theta = np.pi/2
        threshold = 1
        min_line_length = 5
        max_line_gap = 1

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(orig_img,(x1,y1),(x2,y2),(0,0,255),1)

    save(orig_img,'output.png')





