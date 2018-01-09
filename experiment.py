import sys,os,json,argparse
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
    y1 = scan_dim(im,0)
    y1 = butter_highpass_filter(y1,1,25)
    y2 = scan_dim(im,1)
    y2 = butter_highpass_filter(y2,1,25)
    #print('dim is ',len(y1),'long')
    #plt.subplot(211,title='columns collapsed')
    #plt.plot(y1)
    #plt.plot([np.mean(y1)]*len(y1))
    #plt.subplot(212,title='rows collapsed') # creates 2nd subplot with yellow background
    #plt.plot(y2)
    #plt.plot([np.mean(y2)]*len(y2))
    #save(im,'output.png')
    #plt.show()
    #sys.exit(0)
    y1 = y1 > (im.shape[0]*.06)
    y2 = y2 > (im.shape[1]*.06)



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
    corners = []
    lines = []

    # intersects over black pixel
    for i,x in enumerate(xlocs):
        for j,y in enumerate(ylocs):
            if arr[y,x] == 0:
                potential_corners.append((x,y))
                corner_map[i,j] = 1

    # neighboring black pixel intersects
    for y in range(0,len(ylocs)):
        for x in range(0,len(xlocs)-1):
            if corner_map[x,y] and corner_map[x+1,y]:
                l = np.array([(xlocs[x],ylocs[y]),(xlocs[x+1],ylocs[y])])
                potential_lines.append(l)

                c1,c2 = trace_sum(arr,l)
                if c1 == c2:
                    lines.append(l)
                    line_map_h[x,y] = 1

    for y in range(0,len(ylocs)-1):
        for x in range(0,len(xlocs)):
            if corner_map[x,y] and corner_map[x,y+1]:
                l = np.array([(xlocs[x],ylocs[y]),(xlocs[x],ylocs[y+1])])
                potential_lines.append(l)

                c1,c2 = trace_sum(arr,l)
                if c1 == c2:
                    lines.append(l)
                    line_map_v[x,y] = 1

    # detect the corners
    for y in range(0,len(ylocs)-1):
        for x in range(0,len(xlocs)-1):
            if line_map_h[x,y]:
                if line_map_v[x,y]:
                    p1 = (xlocs[x],ylocs[y+1])
                    p2 = (xlocs[x],ylocs[y])
                    p3 = (xlocs[x+1],ylocs[y])
                    corners.append([p1,p2,p3])

                if line_map_v[x+1,y]:
                    p1 = (xlocs[x],ylocs[y])
                    p2 = (xlocs[x+1],ylocs[y])
                    p3 = (xlocs[x+1],ylocs[y+1])
                    corners.append([p1,p2,p3])

    for y in range(len(ylocs)-2,-1,-1):
        for x in range(len(xlocs)-2,-1,-1):
            if line_map_h[x,y+1]:
                if line_map_v[x+1,y]:
                    p1 = (xlocs[x+1],ylocs[y])
                    p2 = (xlocs[x+1],ylocs[y+1])
                    p3 = (xlocs[x],ylocs[y+1])
                    corners.append([p1,p2,p3])

                if line_map_v[x,y]:
                    p1 = (xlocs[x+1],ylocs[y+1])
                    p2 = (xlocs[x],ylocs[y+1])
                    p3 = (xlocs[x],ylocs[y])
                    corners.append([p1,p2,p3])


    corners = np.array(corners)

    for i,x in enumerate(xlocs):
        orig[:,x] = [255,255,0]

    for i,x in enumerate(ylocs):
        orig[x,:] = [255,255,0]


    for x,y in intersects:
        orig[y,x] = [0,0,255]

    for p in potential_corners:
        cv2.circle(orig,p,10,(255,0x8c,0),2 )
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




