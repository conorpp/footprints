import sys,os,json,argparse
from PIL import Image, ImageDraw

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
    ## OCR is pretty greedy so still consider it for lines
    leftover += ocr
    ##

    analyze_circles(leftover)
    circles,leftover = pass_circles(leftover)




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




