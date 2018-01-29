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

    first_rect = rect['contour']
    first_confs = rect['conf']
    outside = rect['ocontour']
    img = rect['img']
    offset = rect['offset']

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

    rect['filled-rects'] = allrects
    rect['scratch-pad'] = scratchpad

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

    rect['features'] = features

    ###
    #for line in lineset:
        #cv2.drawContours(orig,[line],0,[255,0,0],1, )
        #print('contour has %d sides' % ((len(newc)+1)/2))


    #save(orig,'output2.png')
    ###
def line_conf(img,side):
    s,l = line_sum(img, side), line_len(side)
    if l == 0:
        return 0.0
    else:
        return (1.0 - float(s)/l)

def shift_circle(img, circle, inc, conf):
    pt,r = circle
    while circle_confidence(img, (pt,r)) > conf:
        r += inc
        if r == 0:
            return (pt,r)

    return (pt,r)


def irreg_outsides(irregs):
    outsides = []
    for x in irregs:
        img = x['img']
        outside = np.copy(img)
        out_rects = [get_outer_rect(img,r,rect_confidence(img,r)*.9) for r in x['filled-rects']]
        
        if len(out_rects):
            for squ in out_rects:
                #cv2.fillPoly(outside, [squ], 255)
                outside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255
        else:
            squ = get_outer_rect(img,x['contour'])
            outside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255

        for f in x['features']:
            if f[0] == 'circle':
                cir = shift_circle(img, f[1], 1, .1)
                cv2.circle(outside,cir[0],cir[1],255,-1)

        outside = wrap_image(outside,x)
        outsides.append(outside)
    return outsides

def irreg_insides(irregs):
    insides = []
    for x in irregs:
        img = x['img']
        inside = np.zeros(img.shape, dtype=np.uint8)
        in_rects = [get_inner_rect(img,r,rect_confidence(img,r)*.9) for r in x['filled-rects']]

        if len(in_rects):
            for squ in in_rects:
                inside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255
        else:
            squ = get_inner_rect(img,x['contour'])
            inside[squ[2][1]:squ[0][1]+1, squ[2][0]:squ[0][0]+1] = 255

        for f in x['features']:
            if f[0] == 'circle':
                cir = shift_circle(img, f[1], -1, .1)
                cv2.circle(inside,cir[0],cir[1],255,-1)

        inside = (cv2.bitwise_and(img, inside) + (inside != 255) * 255).astype(np.uint8)

        inside = wrap_image(inside,x)
        insides.append(inside)

    return insides

def separate_irregs(inp):
    outsides = []
    outsides += irreg_insides(inp)
    outsides += irreg_outsides(inp)
    return outsides




def move_features_inside(img,features):
    for x in features:
        if x[0] == 'line':
            conf = x[3]
            side = x[1]
            dim,direc = x[2]
            while conf > .4:
                side[0][dim] += direc
                side[1][dim] += direc
                s,l = line_sum(img, side), line_len(side)
                conf = line_conf(img,side)
        else:
            raise ValueError('Invalid feature ' + x[0])

def analyze_irregs(irregs):
    for x in irregs:
        fill_with_rects(x)

def pass_irregs(irregs):
    good = []
    notgood = []
    for x in irregs:
        bad = False
        for f in x['features']:
            if f[0] == 'line':
                if f[3] < .95:
                    bad = True
            elif f[0] == 'circle':
                if f[3] < .45:
                    bad = True
        if not bad:
            good.append(x)
        else:
            notgood.append(x)
    return good,notgood

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
    irregs = []

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
    semir,leftover = pass_rectangles(leftover,0)

    analyze_semi_rects(semir)

    semir,l = pass_semi_rectangles(semir)
    make_irregular_shapes(semir)

    analyze_irregs(l)
    semir2,l = pass_irregs(l)
    print(len(semir2),'irregular rects')
    outs = separate_irregs(semir2+semir)
    outs = block_dots(outs)
    analyze_rectangles(outs)
    leftover += outs
    #outside_semir2 = irreg_outsides(semir2)
    #for x in outside_semir2:
        #save(x,'output2.png')
    #for x in semir2:
        #move_features_inside(x['img'], x['features'])

    irregs += semir
    irregs += semir2
    leftover += l
    #


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
            'irregs': irregs,
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




