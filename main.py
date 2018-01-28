import sys,os,json,argparse
from PIL import Image, ImageDraw
from collections import deque

import numpy as np
import cv2

import analyzers
from utils import *
from filters import *
from analyzers import *
from processors import *
from cli import *
import preprocessing

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

def init(input_file):
    arr = load_image(input_file)
    arr = remove_alpha(arr)
    orig = np.copy(arr)
    arr = polarize(arr)
    analyzers.init(arr)

    arr = preprocessing.preprocess(arr)
    arr = wrap_image(arr)
    return arr,orig

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
    pt = [int(round(pt[0])), int(round(pt[1]))]

    return pt,r



def analyze_semi_rects(orig,rects):
    orig = color(polarize(np.copy(orig)))


    # right, top, left, bottom
    # dim, dir
    lut = [(0,1), (1,-1), (0,-1), (1,1)]

    for x in rects:

        # right, top, left, bottom
        for i,val in enumerate(x['conf']):
            if val > .94: # TODO centralize this value
                continue
            rect = x['contour']
            outside = x['ocontour']
            side = rect[0+i:2+i]
            dim,direc = lut[i]
            
            pt,r = grow_semi_circle(outside,side,dim,direc)

            cconf = circle_confidence(x['img'], (tuple(pt),r))

            # there's half a circle there, stop
            if cconf > .45:
                #print('circle conf',cconf)
                pt[0] += x['offset'][0]
                pt[1] += x['offset'][1]
                cv2.circle(orig,tuple(pt),r,(255,0x8c,0),1 )
                pt[0] -= x['offset'][0]
                pt[1] -= x['offset'][1]

                break

        if cconf > .45:
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

            x['contour'] = np.array(newrect)
            x['conf'] = rect_confidence(x['img'], x['contour'])

        x['semi-circle-conf'] = cconf
        x['semi-circle'] = (pt,r)


    save(orig,'output2.png')

def main():
    args = arguments()

    arr,orig = init(args.input_file)
    print(arr['img'].shape)

    triangles = []
    ocr = []
    lines = []
    rectangles = []
    circles = []
    leftover = []

    submaps = extract_features([arr])
    submaps = block_clipped_components(submaps)

    analyze_rectangles(submaps)

    r,l = pass_rectangles(submaps)
    rectangles += r
    leftover += l

    ## OCR
    analyze_ocr(leftover)
    ocr, leftover = pass_ocr(leftover)

    # check orientations
    rotate_right(leftover)
    analyze_ocr(leftover)
    ocr2, leftover = pass_ocr(leftover)
    ocr += ocr2

    rotate_left(ocr2)
    rotate_left(leftover)
    #
    ## OCR is pretty greedy so still consider it for everything else
    leftover += ocr
    ##

    analyze_circles(leftover)
    circles,leftover = pass_circles(leftover)


    # semi-rects
    semir,l = pass_rectangles(leftover,0)
    leftover = l
    analyze_semi_rects(orig,semir)
    semir,l = pass_semi_rectangles(semir)
    rectangles += semir
    leftover += l

    ##


    # greedily churn out the lines
    while True:
        analyze_lines(leftover)
        r,l = pass_lines(leftover)
        lines += r
        leftover += l

        newlines,leftover = find_line_features(leftover)
        if len(newlines) == 0:
            break
        print(len(newlines),'new lines')

        line_submaps = extract_features(newlines)
        assign_best_fit_lines(line_submaps)
        lines += line_submaps

        leftover = extract_features(leftover)
        leftover = block_dots(leftover)

        analyze_rectangles(leftover)



    analyze_triangles(leftover)
    triangles,leftover = pass_triangles(leftover, arr)



    polish_rectangles(rectangles)
    outs = {'triangles': triangles,
            'ocr': ocr,
            'lines': lines,
            'rectangles': rectangles,
            'circles': circles,
            'leftover': leftover}

    do_outputs(orig,outs)




if __name__ == '__main__':
    main()
    sys.exit(0)

    args = arguments()

    arr = load_image(args.input_file)
    arr = remove_alpha(arr)

    orig = np.copy(arr)
    arr = polarize(arr)

    analyzers.init(arr)

    arr = preprocessing.preprocess(arr)
    arr = wrap_image(arr)

    print(arr['img'].shape)

    submaps = extract_features([arr])

    submaps = block_clipped_components(submaps)

    analyze_rectangles(submaps)


    rectangles,leftover = pass_rectangles(submaps)

    outsides = separate_rectangles(rectangles)
    submaps = extract_features(outsides)

    outsides = block_dots(submaps)
    analyze_rectangles(outsides)

    new_rectangles, new_leftover = pass_rectangles(outsides)
    leftover += new_leftover

    while len(new_rectangles):
        rectangles += new_rectangles

        outsides = separate_rectangles(new_rectangles)
        submaps = extract_features(outsides)

        outsides = block_dots(submaps)
        analyze_rectangles(outsides)

        new_rectangles, new_leftover = pass_rectangles(outsides)
        leftover += new_leftover



    analyze_lines(leftover)
    lines,leftover = pass_lines(leftover)

    potential_lines,leftover = pass_potential_lines(leftover)
    snapshot_imgs(potential_lines,'after potential line pass')


    it = 0
    print('fresh')
    while len(potential_lines):
        while len(potential_lines):
            newitems = separate_lines(potential_lines)
            snapshot_imgs(newitems,'after line separation')
            it += 1
            submaps = extract_features(newitems)

            submaps = block_dots(submaps)
            snapshot_imgs(submaps,'after extraction')

            #trim_images(submaps)
            snapshot_imgs(submaps,'after trim')
            analyze_rectangles(submaps)
            analyze_lines(submaps)
            lines2,leftover2 = pass_lines(submaps)
            print('pass %d.  found %d lines and %d more leftover' % (it,len(lines2), len(leftover2)))
            potential_lines,leftover2 = pass_potential_lines(leftover2)
            snapshot_imgs(potential_lines,'passed for potential line')
            print('there\'s %d possible lines and %d not containing lines' % (len(potential_lines), len(leftover2)))


            leftover += leftover2

            if len(lines2) == 0:
                print('no more lines to find')
                leftover += potential_lines
                break

            lines += lines2

        new_rectangles, leftover = pass_rectangles(leftover)
        potential_lines = []
        while len(new_rectangles):
            rectangles += new_rectangles

            outsides = separate_rectangles(new_rectangles)
            submaps = extract_features(outsides)

            outsides = block_dots(submaps)
            analyze_rectangles(outsides)

            new_rectangles, new_leftover = pass_rectangles(outsides)
            potential_lines += new_leftover

        analyze_lines(potential_lines)


    analyze_triangles(leftover)
    triangles,leftover = pass_triangles(leftover)

    analyze_circles(leftover)
    circles,leftover = pass_circles(leftover)

    analyze_ocr(leftover)
    ocr, leftover = pass_ocr(leftover)

    # check orientations
    rotate_right(leftover)
    analyze_ocr(leftover)
    ocr2, leftover = pass_ocr(leftover)
    ocr += ocr2

    rotate_left(leftover)
    #

    # remove slashes
    slashes, ocr = pass_slashes(ocr)
    leftover += slashes


    polish_rectangles(rectangles)

    #potential_lines,_ = pass_potential_lines(leftover)

    t1 = timestamp()
    #potential_lines
    newlines,lineleftover = find_line_features(leftover)
    scanned_lines = leftover
    #scanned_lines = []

    line_submaps = extract_features(newlines)
    assign_best_fit_lines(line_submaps)
    lines += line_submaps

    leftover = block_dots(lineleftover)
    leftover = extract_features(leftover)
    analyze_rectangles(leftover)


    newlines,leftover = find_line_features(leftover)
    line_submaps = extract_features(newlines)
    assign_best_fit_lines(line_submaps)
    lines += line_submaps

    #leftover = block_dots(leftover)
    leftover = extract_features(leftover)
    analyze_rectangles(leftover)


    newlines,leftover = find_line_features(leftover)
    line_submaps = extract_features(newlines)
    assign_best_fit_lines(line_submaps)
    lines += line_submaps

    leftover = extract_features(leftover)
    leftover = block_dots(leftover)
    analyze_rectangles(leftover)


    analyze_triangles(leftover)
    triangles2,leftover = pass_triangles(leftover)
    triangles += triangles2

    outs = {'triangles': triangles,
            'ocr': ocr,
            'lines': lines,
            'rectangles': rectangles,
            'circles': circles,
            'leftover': leftover}

    do_outputs(orig,outs)




