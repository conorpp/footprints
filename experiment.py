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



def timestamp(): return int(round(time.time() * 1000))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    arr = remove_alpha(arr)

    #arr[160,:] = 255
    #arr[int(763 * .50),:] = 255
    #arr[int(763 * .65),:] = 255
    #arr[int(763 * .95),:] = 255

    #arr[:,int(802 * .97)] = 255



    orig = np.copy(arr)
    arr = polarize(arr)
    im = np.copy(arr)
    #arr = wrap_image(arr)
    
    #output = cv2.connectedComponentsWithStats((arr['img']), 4)
    #trim_images([arr])

    #analyzers.init(arr)
    #print(arr['img'].shape)

    # detect circles in the image
    #circles = cv2.HoughCircles(arr['img'], cv2.HOUGH_GRADIENT, 1, 50,
            #param1=255, param2=25, minRadius=0, maxRadius=100)

    #lines = cv2.HoughLinesP(255-arr['img'], 1, np.pi/180, 150, 35)
    t1 = timestamp()
    y1 = scan_dim(im,0)
    y1mean = np.mean(y1)
    y1 = butter_highpass_filter(y1,1,25)
    y2 = scan_dim(im,1)
    y2mean = np.mean(y2)
    y2 = butter_highpass_filter(y2,1,25)
    t2 = timestamp()
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
    y1 = y1 > (im.shape[0]*.06)
    y2 = y2 > (im.shape[1]*.06)
    print('filter time: %d ms' % (t2-t1))



    #submaps = extract_features([arr])
    #submaps = extract_components([arr])

    #print('before clipped block',len(submaps))
    #submaps = block_clipped_components(submaps)
    #trim_images(submaps)
    #print('after',len(submaps))
    #trim_images(submaps)

    #analyze_rectangles(submaps)

    #rectangles,leftover = pass_rectangles(submaps)


    #polish_rectangles(rectangles)


    xlocs = np.where(y1)[0]
    xlocs = np.split(xlocs, np.where(np.diff(xlocs) != 1)[0]+1)
    xlocs = [x[int(round(len(x)/2.))] for x in xlocs]

    ylocs = np.where(y2)[0]
    ylocs = np.split(ylocs, np.where(np.diff(ylocs) != 1)[0]+1)
    ylocs = [x[int(round(len(x)/2.))] for x in ylocs]

    corner_map = np.zeros((len(xlocs),len(ylocs)))

    line_map_h = np.zeros((len(xlocs)-1,len(ylocs)))    # dual of corner map
    line_map_v  = np.zeros((len(xlocs),len(ylocs)-1))    # dual of corner map

    intersects = [[x,y] for x in xlocs for y in ylocs]

    potential_corners = []
    potential_lines = []
    tlcorners = []
    trcorners = []
    blcorners = []
    brcorners = []
    rectangles = []
    lines = []



    # intersects over black pixel
    t1 = timestamp()
    for i,x in enumerate(xlocs):
        for j,y in enumerate(ylocs):
            if arr[y,x] == 0:
                potential_corners.append((x,y))
                corner_map[i,j] = 1

    # neighboring black pixel intersects
    for y in range(0,len(ylocs)):
        for x in range(0,len(xlocs)-1):
            if corner_map[x,y] and corner_map[x+1,y]:
                l = [(xlocs[x],ylocs[y]),(xlocs[x+1],ylocs[y])]
                potential_lines.append(l)

                if line_exists(arr,l):
                    lines.append(l)
                    line_map_h[x,y] = 1

    for y in range(0,len(ylocs)-1):
        for x in range(0,len(xlocs)):
            if corner_map[x,y] and corner_map[x,y+1]:
                l = [(xlocs[x],ylocs[y]),(xlocs[x],ylocs[y+1])]
                potential_lines.append(l)

                if line_exists(arr,l):
                    lines.append(l)
                    line_map_v[x,y] = 1

    t2 = timestamp()
    print('lines time: %d ms' % (t2-t1))

    t1 = timestamp()

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

    t2 = timestamp()
    print('corners time: %d ms' % (t2-t1))

    t1 = timestamp()

    # detect the rectangles
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

    t2 = timestamp()
    print('rectangles time: %d ms' % (t2-t1))


    corners = blcorners + brcorners + tlcorners + trcorners
    corners = np.array(corners)
    rectangles = sorted(rectangles,key = lambda x: rect_area(x))
    #rectangles = sorted(rectangles,key = lambda x: rect_area(x), reverse=True)
    rect_map = np.zeros(arr.shape)+1
    rectangle_filter = []
    overlapping_rects = []

    # block redundant rectangles and detect overlapping rects
    rectangles = np.array(rectangles)
    t1 = timestamp()

    for x in rectangles:
        count = 0
        for i in range(0,4):
            side = x[0+i:2+i]
            if line_exists(rect_map,side):
                count += 1
                last_side = (i,2+i)
            else:
                set_line(rect_map,side)
        if count <= 1:
            if count: overlapping_rects.append([x,last_side])
            else:
                rectangle_filter.append(x)

    rectangles = rectangle_filter
    rectangle_filter = []
    overlap_pairs = []
    rejects = []

    # get the offending pair for each rect
    for r in overlapping_rects:
        rect = r[0]
        i = r[1]
        side = rect[i[0]:i[1]]
        for r2 in rectangles:
            if (side == r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2]).all():
                overlap_pairs.append((rect,r2))
                rectangles.remove(r2)
            elif (side == np.flip(r2[(i[0] + 2) % 4:(i[1] ) % 4 + 2],0)).all():
                overlap_pairs.append((rect,r2))
                rectangles.remove(r2)

    t2 = timestamp()
    print('overlaps time: %d ms' % (t2-t1))


    
    t1 = timestamp()

    inv = (arr==0)
    r1means = np.zeros(4)
    r2means = np.zeros(4)
    r1vars = np.zeros(4)
    r2vars = np.zeros(4)
    for r1,r2 in overlap_pairs:
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

    t2 = timestamp()
    print('find rejects time: %d ms' % (t2-t1))


    print(len(rejects),' rejects')


    for i,x in enumerate(xlocs):
        orig[:,x] = [255,255,0]

    for i,x in enumerate(ylocs):
        orig[x,:] = [255,255,0]


    for x,y in intersects:
        orig[y,x] = [0,0,255]

    #for p in potential_corners:
        #cv2.circle(orig,p,10,(255,0x8c,0),2 )
    for l in potential_lines:
        cv2.line(orig,tuple(l[0]),tuple(l[1]), (255,200,0),1)

    print(len(lines),' solid lines')
    for l in lines:
        cv2.line(orig,tuple(l[0]),tuple(l[1]), (0,255,0),1)
    
    print(len(corners),'solid corners')
    for c in corners:
        cv2.line(orig,tuple(c[0]),tuple(c[1]), (0,255,255),1)
        cv2.line(orig,tuple(c[1]),tuple(c[2]), (0,255,255),1)
        cv2.circle(orig,tuple(c[1]),10,(255,0,255),2 )


    print(len(rectangles),'rectangles')
    for i,r in enumerate(rectangles):
        cv2.drawContours(orig,[r],0,[255,0,255],2)

    for i,r in enumerate(rejects):
        cv2.drawContours(orig,[np.array(r)],0,[255,0,0],2)
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




