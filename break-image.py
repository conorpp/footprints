import sys,os,json,argparse
from PIL import Image, ImageDraw

import numpy as np


if len(sys.argv) != 3:
    print 'usage: %s <input.png> <output>' % sys.argv[0]
    sys.exit(1)

im = Image.open(sys.argv[1])
output = sys.argv[2]

arr = np.array(im)

# x: row, y: column, z:rgba, origin is upper-left

# polarize
for i in range(0,arr.shape[0]):
    for j in range(0, arr.shape[1]):
        if sum(arr[i][j]) < (255 + 128*3):
            arr[i][j][0] = 0
            arr[i][j][1] = 0
            arr[i][j][2] = 0
        else:
            arr[i][j][0] = 255
            arr[i][j][1] = 255
            arr[i][j][2] = 255
print 'begin tracking'

def explore(arr,i,j):
    trackmap = np.zeros(arr.shape[:2])
    return explore_r(arr,i,j,trackmap)

def explore_r(arr,i,j,trackmap):
    imax = arr.shape[0]-1
    jmax = arr.shape[1]-1

    nodes_to_visit = [(i,j)]

    def checkout(i,j):
        #print i,j
        if arr[i][j][0] == 0:
            if not trackmap[i,j]:
                trackmap[i,j] = 1             # add to list
                nodes_to_visit.append((i,j))

    while len(nodes_to_visit):
        node = nodes_to_visit.pop()
        trackmap[node[0],node[1]] = 1

        i = node[0]
        j = node[1]
        arr[i,j,2] = 128              # debug it

        # right
        if (j+1) <= jmax:
            checkout(i,j+1)

            # bottom-right
            if (i+1) <= imax:
                checkout(i+1,j+1)

            # top-right
            if (i-1) >= 0:
                checkout(i-1,j+1)

        # left
        if (j-1) >= 0:
            checkout(i,j-1)

            # bottom-right
            if (i+1) <= imax:
                checkout(i+1,j-1)

            # top-right
            if (i-1) >= 0:
                checkout(i-1,j-1)

        # top
        if (i - 1) >= 0:
            checkout(i-1,j)

        # bottom
        if (i + 1) <= imax:
            checkout(i+1,j)

    return trackmap

def apply_mapping(im,mapping):
    cpy = np.copy(im)
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            if mapping[i,j]:
                cpy[i,j,0] = 255
                cpy[i,j,1] = 1
                cpy[i,j,2] = 1
    return cpy

def gen_image(mapping):
    cpy = np.zeros((mapping.shape[0], mapping.shape[1],3), dtype = np.uint8) + 255
    for i in range(0,cpy.shape[0]):
        for j in range(0,cpy.shape[1]):
            if mapping[i,j]:
                cpy[i,j,0] = 0
                cpy[i,j,1] = 0
                cpy[i,j,2] = 0
    return cpy

for runs in range(300,350+1):
    done = 0
    track_map = np.zeros(arr.shape[:2])
    submaps = [0]
    stop = 0

    try:
        for i in range(0,arr.shape[0]):
            for j in range(0,arr.shape[1]):
                if arr[i][j][0] == 0:
                    if not track_map[i,j]:
                        print 'exploring..'
                        track_map[i,j] = 1
                        submap = explore(arr,i,j)
                        track_map += submap
                        submaps.append(submap)
                        done += 1
                if done >= runs: break
            if done >= runs: break
    except RuntimeError as e:
        print e
        stop = 1

    if done < runs: stop = 1

    print 'writing out submappings..'
    for i,x in enumerate(submaps[1:]):
        print i
        out = gen_image(x)

        out = Image.fromarray(out)
        out.save(output + ('/item%d.png' % i))

    if stop: break

