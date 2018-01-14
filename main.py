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

    rects2 = pass_rectangles(ocr)
    print('rects 2:',len(rects2))


    polish_rectangles(rectangles)

    potential_lines,_ = pass_potential_lines(leftover)
    navigate_lines(potential_lines)
    print(len(potential_lines),'potential lines left')


    for x in (lines + rectangles + triangles + ocr):
        x['cl'] = True
    for x in (leftover):
        x['cl'] = False
    if 0:
        for x in sorted(leftover+lines+ocr, key=lambda x: x['id']):
            print_img(x)
            cpy = np.copy(orig)
            cv2.drawContours(cpy,[x['line']],0,[255,0,0],1,offset=tuple(x['offset']))
            encircle(cpy, x['line'], offset=x['offset'])

            xx,yy,w,h = cv2.boundingRect(x['ocontour'])
            [xx,yy] = xx+x['offset'][0],yy+x['offset'][1]
            cv2.rectangle(cpy,(xx,yy),(xx+w,yy+h),(0,0,255),2)
            postfix = 'C' if x['cl'] else 'U'
            save(cpy,'out/line%c%d.png' % (postfix,x['id']))
            #save(x,'out/item%c%d.png' % (postfix,x['id']))


    print('%d rectangles' % len(rectangles))
    for x in rectangles:
        cv2.drawContours(orig,[x['rectangle']],0,[255,0,255],1, offset=tuple(x['offset']))
        save(x['img'],'out/rect%d.png' % (x['id']))

    print('%d lines' % len(lines))
    for x in lines:
        cv2.drawContours(orig,[x['line']],0,[0,128,0],2, offset=tuple(x['offset']))

    print('%d triangles' % len(triangles))
    for x in triangles:
        cv2.drawContours(orig,[x['triangle']],0,[0,0,255],1, offset=tuple(x['offset']))

    print('%d characters' % len(ocr))
    for x in ocr:
        cv2.drawContours(orig,[x['ocontour']],0,[0,255,255],2, offset=tuple(x['offset']))
        #print('%d == %s (%d%%)' % (x['id'],x['symbol'],x['ocr-conf']))
        #save(x,'out/ocr%d.png' % x['id'])

    print('%d circles' % len(circles))
    for x in circles:
        off = x['offset']
        xp = off[0] + x['circle'][0][0]
        yp = off[1] + x['circle'][0][1]
        cv2.circle(orig,(xp,yp),x['circle'][1],(255,0x8c,0),2 )

    print('%d unclassified items' % len(leftover))
    for x in leftover:
        if contains_line(x):
            cv2.drawContours(orig,[x['ocontour']],0,[255,255,0],1, offset=tuple(x['offset']))
            if x['target']:
                for t in x['traces']:
                    #print(t)
                    p1x = t[0][0] + x['offset'][0]
                    p1y = t[0][1] + x['offset'][1]
                    p2x = t[1][0] + x['offset'][0]
                    p2y = t[1][1] + x['offset'][1]
                    cv2.line(orig,tuple([p1x,p1y]),tuple([p2x,p2y]),[253,0x8c,0],1)


        else:
            cv2.drawContours(orig,[x['ocontour']],0,[255,0,0],1, offset=tuple(x['offset']))
            cv2.drawContours(orig,[x['triangle']],0,[255,0,128],1, offset=tuple(x['offset']))

    save(orig,'output.png')
    for x in sorted(leftover + rectangles + lines + triangles + ocr, key = lambda x:x['id']):

        #if x['id'] == 497:
        #if x in triangles:
            #save_history(x)
        #if x in rectangles:
            #save_history(x)
        #print('saving %d' % (x['id'],) )
        #save(x,'out/item')
        pass




