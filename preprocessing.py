import sys,os,json,argparse,time

from scipy import signal
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

def line_exists(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    s = np.sum(arr[y1:y2+1,x1:x2+1])

    return s == 0

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


def set_line(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    arr[y1:y2+1,x1:x2+1] = 0

def rect_area(r):
    #r = [tl[1],tr[1],br[1],bl[1],tl[1]]
    dx = abs(r[0][0] - r[2][0])
    dy = abs(r[0][1] - r[2][1])
    return dx * dy



def get_intersects(y1,y2,margin1,margin2):
    y1 = y1 > margin1
    y2 = y2 > margin2
    xlocs = np.where(y1)[0]
    xlocs = np.split(xlocs, np.where(np.diff(xlocs) != 1)[0]+1)
    xlocs = [x[int(round(len(x)/2.))] for x in xlocs]

    ylocs = np.where(y2)[0]
    ylocs = np.split(ylocs, np.where(np.diff(ylocs) != 1)[0]+1)
    ylocs = [x[int(round(len(x)/2.))] for x in ylocs]
    return xlocs,ylocs

def get_line_signals(im):
    y1 = scan_dim(im,0)
    y1mean = np.mean(y1)
    y1 = butter_highpass_filter(y1,1,25)
    y2 = scan_dim(im,1)
    y2mean = np.mean(y2)
    y2 = butter_highpass_filter(y2,1,25)

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


    return y1,y2

def get_corner_map(arr, xlocs,ylocs):
    corner_map = np.zeros((len(xlocs),len(ylocs)))
    for i,x in enumerate(xlocs):
        for j,y in enumerate(ylocs):
            if arr[y,x] == 0:
                corner_map[i,j] = 1
    return corner_map


def get_line_map(arr,corner_map,xlocs,ylocs):
    line_map_h = np.zeros((len(xlocs)-1,len(ylocs)))    # dual of corner map
    line_map_v  = np.zeros((len(xlocs),len(ylocs)-1))    # dual of corner map


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


def get_rectangles(arr,tlcorners, trcorners, brcorners, blcorners):
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


def detect_overlap(arr,rectangles):
    skips = 0
    rect_map = np.zeros(arr.shape)+1
    rects = []
    overlaps = []
    for x in rectangles:
        count = 0
        for i in range(0,4):
            side = x[0+i:2+i]
            if line_exists(rect_map,side):
                count += 1
                last_side = (i,2+i)
            else:
                set_line(rect_map,side)
        #if count <= 1:
        if count: 
            np_rm(rectangles,x)
            #print('last_side: ', last_side2)
            return [x,last_side]
        #else:
            #np_rm(rectangles,x)
            #skips += 1
            #print('dump overlap')
            #pass
    #print('blocked',skips,' overlapping rectangles')
    #return rects, overlaps

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

def take_match(rectangles, overlap):

    r = overlap
    rect = r[0]
    i = r[1]
    side = rect[i[0]:i[1]]
    #print ((i[0] ) ,(i[1] ) )
    #print ((i[0] + 2) % 1,(i[1] ) % 4 + 2)
    #print('side',side)
    for r2 in rectangles:
        #r2 = r2[:4]
        #print(len(r2))
        #assert(len(r2) == 4)
        if (side == r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2]).all():
            np_rm(rectangles,r2)
            return (rect,r2)
            #overlapping_rects.remove(r)
            #np_rm(overlapping_rects,r)
            #rectangles.remove(r2)
        elif (side == np.flip(r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2],0)).all():
            np_rm(rectangles,r2)
            return (rect,r2)
            #overlapping_rects.remove(r)
            #np_rm(overlapping_rects,r)
            #rectangles.remove(r2)


def reconcile_overlaps(arr,rectangles,overlap_pair):
    rejects = []
    inv = (arr==0)
    r1means = np.zeros(4)
    r2means = np.zeros(4)
    r1vars = np.zeros(4)
    r2vars = np.zeros(4)
    r1,r2 = overlap_pair
    for i in range(1,10):
        r1t = [line_trace(inv,r1[x:x+2],i,x&1) for x in range(0,4)]
        r2t = [line_trace(inv,r2[x:x+2],i,x&1) for x in range(0,4)]

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

        if rej is not None:
            rejects.append(rej)
            if rej is r1:
                rectangles.append(r2)
            else:
                rectangles.append(r1)
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
        tl = vminx[np.where(vminx_y == np.max(vminx_y))][0]
        bl = vminx[np.where(vminx_y == np.min(vminx_y))][0]
        
        vmaxx_y = vmaxx[:,1]
        tr = vmaxx[np.where(vmaxx_y == np.max(vmaxx_y))][0]
        br = vmaxx[np.where(vmaxx_y == np.min(vmaxx_y))][0]

        print('merged rect: ', tl, tr, br, bl)
        rectangles.append(np.array([tl,tr,br,bl,tl]))
 
    return rejects


