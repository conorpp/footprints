import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np
import cv2

from utils import *
from filters import *
from analyzers import *
from processors import *


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input.png>' % sys.argv[0])
        sys.exit(1)

    arr = load_image(sys.argv[1])
    arr = remove_alpha(arr)

    orig = np.copy(arr)
    arr = polarize(arr)
    arr = wrap_image(arr)

    submaps = extract_components([arr])
    snapshot_imgs(submaps,'extract_components parent',arr)
    snapshot_imgs(submaps,'before trim')

    trim_images(submaps)
    snapshot_imgs(submaps,'after trim')

    analyze_rectangles(submaps)
    #snapshot_imgs(rectangles,'after analyze_rectangles')

    rectangles,leftover = pass_rectangles(submaps)

    outsides = separate_rectangles(rectangles)
    submaps = extract_components(outsides)

    outsides = block_dots(submaps)
    analyze_rectangles(outsides)
    leftover += outsides
    #snapshot_imgs(rectangles,'im a rectangle')
    #snapshot_imgs(leftover,'im not a rectangle')

    analyze_lines(leftover)

    #snapshot_imgs(lines,'after analyze_lines')

    lines,leftover = pass_lines(leftover)
    
    potential_lines,leftover = pass_potential_lines(leftover)
    snapshot_imgs(potential_lines,'after potential line pass')


    it = 0
    print('fresh')
    while len(potential_lines):
        newitems = separate_lines(potential_lines)
        snapshot_imgs(newitems,'after line separation')
        it += 1
        submaps = extract_components(newitems)

        submaps = block_dots(submaps)
        snapshot_imgs(submaps,'after extraction')

        trim_images(submaps)
        snapshot_imgs(submaps,'after trim')
        analyze_rectangles(submaps)
        analyze_lines(submaps)
        lines2,leftover2 = pass_lines(submaps)
        print('pass %d.  found %d lines and %d more leftover' % (it,len(lines2), len(leftover2)))
        potential_lines,leftover2 = pass_potential_lines(leftover2)
        snapshot_imgs(potential_lines,'passed for potential line')
        print('there\'s %d possible lines and %d not containing lines' % (len(potential_lines), len(leftover2)))
        #leftover2 += potential_lines
        #break
        lines += lines2
        leftover += leftover2   # not lines
        #break
    
    analyze_triangles(leftover)
    triangles,leftover = pass_triangles(leftover)

    analyze_ocr(leftover)
    ocr, leftover = pass_ocr(leftover)
    #lines += lines2
    #print('%d classified lines.  %d from second pass' % (len(lines),len(lines2)))
    #leftover += leftover2
    #print('%d unclassified items. %d from second pass' % (len(leftover), len(leftover2)))

    for x in (lines + rectangles + triangles + ocr):
        x['cl'] = True
    for x in (leftover):
        x['cl'] = False
    if 1:
        for x in sorted(triangles, key=lambda x: x['id']):
            print_img(x)
            cpy = np.copy(orig)
            cv2.drawContours(cpy,[x['line']],0,[255,0,0],1,offset=tuple(x['offset']))
            encircle(cpy, x['line'], offset=x['offset'])

            xx,yy,w,h = cv2.boundingRect(x['ocontour'])
            [xx,yy] = xx+x['offset'][0],yy+x['offset'][1]
            cv2.rectangle(cpy,(xx,yy),(xx+w,yy+h),(0,0,255),2)
            postfix = 'C' if x['cl'] else 'U'
            #save(cpy,'out/line%c%d.png' % (postfix,x['id']))
            save(x,'out/item%c%d.png' % (postfix,x['id']))


    print('%d rectangles' % len(rectangles))
    for x in rectangles:
        cv2.drawContours(orig,[x['contour']],0,[255,0,255],1, offset=tuple(x['offset']))
        save(x['img'],'out/rect%d.png' % (x['id']))

    print('%d lines' % len(lines))
    for x in lines:
        cv2.drawContours(orig,[x['line']],0,[0,128,0],2, offset=tuple(x['offset']))

    print('%d triangles' % len(triangles))
    for x in triangles:
        cv2.drawContours(orig,[x['triangle']],0,[0,0,255],2, offset=tuple(x['offset']))

    print('%d characters' % len(ocr))
    for x in ocr:
        cv2.drawContours(orig,[x['ocontour']],0,[0,255,255],2, offset=tuple(x['offset']))

    print('%d unclassified items' % len(leftover))
    for x in leftover:
        if contains_line(x):
            cv2.drawContours(orig,[x['ocontour']],0,[255,255,0],1, offset=tuple(x['offset']))
        else:
            cv2.drawContours(orig,[x['ocontour']],0,[255,0,0],1, offset=tuple(x['offset']))

    save(orig,'output.png')
    for x in sorted(leftover + rectangles + lines + triangles + ocr, key = lambda x:x['id']):

        #if x['id'] == 284:
        #if x in triangles:
            #save_history(x)
        #print('saving %d' % (x['id'],) )
        #save(x,'out/item')
        pass




