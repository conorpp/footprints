import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2

import analyzers
from utils import *
from filters import *
from analyzers import *
from processors import *
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

def arguments():
    parser = argparse.ArgumentParser(description='drawing parser',
            epilog='extra annotations: ocontour,brect,rect,line,tri,circle,ocr\n')
    parser.add_argument('input_file')
    parser.add_argument('--all',help='use all features for operation')
    parser.add_argument('--action', action='store',default='display',
            help='specify action to use on output. display[default],binary,redraw,none')
    parser.add_argument('--save-features', action='store_true', dest='save_features',
            help='save features to output files.  Use switches to select which ones.')
    parser.add_argument('-t', action='store_true',help='triangles (for displaying..)')
    parser.add_argument('-r', action='store_true',help='rectangles')
    parser.add_argument('-c', action='store_true',help='circles')
    parser.add_argument('-l', action='store_true',help='lines')
    parser.add_argument('-o', action='store_true',help='OCR')
    parser.add_argument('-x', action='store_true',help='leftovers')

    parser.add_argument('-T', action='store_true',help='triangles (for saving to file..)')
    parser.add_argument('-R', action='store_true',help='rectangles')
    parser.add_argument('-C', action='store_true',help='circles')
    parser.add_argument('-L', action='store_true',help='lines')
    parser.add_argument('-O', action='store_true',help='OCR')
    parser.add_argument('-X', action='store_true',help='leftovers')

    parser.add_argument('--save-type', default='large', action='store', dest='save_type',help='small,large,outlined')
    parser.add_argument('--bg', action='store_true', help='use original image as background for --save-type')
    parser.add_argument('-a', action='append', dest='outa',help='extra annotations to add to output image.')
    parser.add_argument('-b', action='append', dest='savea',help='extra annotations to add to output features.')
    parser.add_argument('-A', action='append', dest='botha',help='extra annotations to add to both output features and image.')

    args = parser.parse_args()
    return args

def do_outputs(orig,outs):
    print('%d triangles' % len(outs['triangles']))
    print('%d rectangles' % len(outs['rectangles']))
    print('%d OCR' % len(outs['ocr']))
    print('%d lines' % len(outs['lines']))
    print('%d leftover' % len(outs['leftover']))

    args = arguments()

    def put_thing(im, x, color, offset=None):
        if offset is not None: offset = tuple(offset)
        cv2.drawContours(im,[x],0,color,1, offset=offset)

    def put_triangle(im, tri, offset=None):
        put_thing(im,tri['triangle'],[0,0,255],offset)

    def put_triangles(im,triangles):
        for x in triangles:
            offset = tuple(x['offset'])
            put_triangle(im,x,offset)
    
    def put_rectangles(im,rectangles):
        for x in rectangles:
            offset = tuple(x['offset'])
            cv2.drawContours(im,[x['rectangle']],0,[255,0,255],1, offset=offset)

    def put_lines(im,lines):
        for x in lines:
            offset = tuple(x['offset'])
            cv2.drawContours(im,[x['line']],0,[0,128,0],2, offset=offset)

    def put_ocr(im,ocr):
        for x in ocr:
            offset = tuple(x['offset'])
            cv2.drawContours(im,[x['ocontour']],0,[0,255,255],2, offset=offset)

    def put_circles(im,circles):
        for x in circles:
            offset = x['offset']
            off = offset
            xp = off[0] + x['circle'][0][0]
            yp = off[1] + x['circle'][0][1]
            cv2.circle(im,(xp,yp),x['circle'][1],(255,0x8c,0),2 )

    def put_leftover(im,circles):
        for x in leftover:
            offset = tuple(x['offset'])
            cv2.drawContours(im,[x['ocontour']],0,[255,0,0],1, offset=offset)

    def save_things(things,bname='im',path='./out'):
        for i,x in enumerate(things):
            n = path +'/' + bname+('%d.png' % (i))
            print('saving ',n)
            save(x,n)
    
    if args.action == 'display':
        pass
    elif args.action == 'binary':
        orig = polarize(orig)
        orig = color(orig)
    elif args.action == 'redraw':
        pass #TODO

    if args.action != 'none':
        if args.t:
            put_triangles(orig, outs['triangles'])
        if args.r:
            put_rectangles(orig, outs['rectangles'])
        if args.c:
            put_circles(orig, outs['circles'])
        if args.l:
            put_lines(orig, outs['lines'])
        if args.o:
            put_ocr(orig, outs['ocr'])
        if args.x:
            put_leftover(orig, outs['leftover'])


    saving_list = []
    if args.T:
        saving_list += outs['triangles']
    if args.R:
        saving_list += outs['rectangles']
    if args.C:
        saving_list += outs['circles']
    if args.L:
        saving_list += outs['lines']
    if args.O:
        saving_list += outs['ocr']
    if args.X:
        saving_list += outs['leftover']

    if args.save_type == 'large' or args.save_type == 'outline':
        bg = np.zeros(orig.shape,dtype=np.uint8)+255
        if args.bg:
            bg = orig
        for x in saving_list:
            newim = np.copy(bg)
            oldim = x['img']
            oj,oi = x['offset']
            for i in range(0,oldim.shape[0]):
                for j in range(0,oldim.shape[1]):
                    #print(oldim)
                    v = oldim[i,j]
                    newim[i+oi,j+oj] = v

            x['img'] = newim
    if args.save_type == 'outline':
        for x in saving_list:
            encircle(x['img'], x['ocontour'], offset=x['offset'])


    if args.save_features:
        if args.T:
            save_things(outs['triangles'], 'tri')
        if args.R:
            save_things(outs['rectangles'], 'r')
        if args.C:
            save_things(outs['circles'], 'cir')
        if args.L:
            save_things(outs['lines'], 'l')
        if args.O:
            save_things(outs['ocr'], 'ocr')
        if args.X:
            save_things(outs['leftover'], 'leftover')


    save(orig,'output.png')



if __name__ == '__main__':

    args = arguments()

    arr = load_image(args.input_file)
    arr = remove_alpha(arr)

    #arr[160,:] = 255
    #arr[int(763 * .50),:] = 255
    #arr[int(763 * .65),:] = 255
    #arr[int(763 * .95),:] = 255

    #arr[:,int(802 * .97)] = 255



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




