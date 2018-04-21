import numpy as np
from utils import line_len, line_slope
from scipy import stats
from rtree import index

from settings import *

class RTree():
    def __init__(self,shape,use_offset=True):
        self.features = {}
        self.tree = index.Index()
        self.shape = shape
        self.use_offset = use_offset
        self.counter = 0
        self.xmax = shape[1]
        self.ymax = shape[0]
        self.xmin = 0
        self.ymin = 0

    def has(self,idx):
        return (idx in self.features)

    def get_bounds(self, x,use_offset=True):
        left,bottom = x['boundxy']
        right,top = left + x['width'], bottom+ x['height']
        if self.use_offset:
            left += x['offset'][0]
            right += x['offset'][0]
            bottom += x['offset'][1]
            top += x['offset'][1]
        return left,bottom,right,top

    def add_obj(self,x,key=None):
        if key is None:
            left,bottom,right,top = self.get_bounds(x,)
        else:
            left,bottom,right,top = key(x)

        if hasattr(x,'id'):
            self.tree.insert(x.id,(left,bottom,right,top))
            self.features[x.id] = x
        else:
            self.tree.insert(self.counter,(left,bottom,right,top))
            self.features[self.counter] = x
            self.counter += 1

        #assert(x['id'] != 265)

    def add_objs(self,objs,key=None):
        for x in objs:
            self.add_obj(x,key)

    def remove(self,idx):
        #if bound is None:
        bound = self.get_bounds(self.features[idx])
        self.tree.delete(idx,bound)
        del self.features[idx]

    def items(self,):
        bound = self.tree.bounds
        return self.tree.intersection(bound)

    def intersect(self, x, factor=0.0):
        l = []
        left,bottom = x['boundxy']
        right,top = left + x['width']*(1 + factor), bottom+ x['height']*(1 + factor)
        left += x['offset'][0]
        right += x['offset'][0]
        bottom += x['offset'][1]
        top += x['offset'][1]

        left -= x['width'] * factor
        bottom -= x['width'] * factor

        ids = self.tree.intersection((left,bottom,right,top))
        for x in ids:
            l.append(self.features[x])
        return l

    def intersectBox(self, box, offset=None):
        l = []
        left,bottom,right,top = box
        #left,bottom = x['boundxy']
        #right,top = left + x['width']*(1 + factor), bottom+ x['height']*(1 + factor)
        if offset is not None:
            left += offset[0]
            right += offset[0]
            bottom += offset[1]
            top += offset[1]

        #left -= x['width'] * factor
        #bottom -= x['width'] * factor

        ids = self.tree.intersection((left,bottom,right,top))
        for x in ids:
            l.append(self.features[x])
        return l


    def intersectPoint(self, pt, sz = 1):
        l = []
        left,bottom,right,top = pt[0] - sz, pt[1] - sz,pt[0] + sz,pt[1] + sz

        ids = self.tree.intersection((left,bottom,right,top))
        for x in ids:
            l.append(self.features[x])
        return l



    def nearest(self, x, num=1):
        l = []
        left,bottom = x['boundxy']
        right,top = left + x['width'], bottom+ x['height']
        left += x['offset'][0]
        right += x['offset'][0]
        bottom += x['offset'][1]
        top += x['offset'][1]

        ids = self.tree.nearest((left,bottom,right,top), num)
        for x in ids:
            l.append(self.features[x])
        return l

    def test_coherency(self,objs):
        items = self.items()
        items = sorted([x for x in items])
        items2 = sorted([x.id for x in objs])
        coher = True
        for x in objs:
            if (x.id in items):
                pass
            else:
                print('error %d is not in tree' % x.id)
                coher = False
        for x in items:
            if x not in items2:
                print('error %d is not in list' % x.id)
                coher = False

        if coher:
            #print('TREE is COHERENT with LIST')
            return True
        else:
            print(items)
            print(items2)
            print(len(items), 'vs', len(objs))
        raise RuntimeError('RTree is not coherent with list')
        #return False

counter = 0
class Shape():
    img = None
    height = 1
    width = 1
    history = None
    comment = ''
    conf = 0
    area_ratio = 0
    a1 = 0
    a2 = 0
    contour = None
    offset = None

    line_conf = 0
    aspect_ratio = 0
    line_length = 0
    length_area_ratio = 0
    vertical = 0

    sum = None
    rotated = False
    line_scan_attempt = 0
    merged = False

    line_estimates = None
    features = None
    traces = None

    trash = False

    symbol = ''

    def __init__(self,im=None,parent=None, offset=None):
        self.history = []
        self.img = im
        self.contour = []
        if parent is None:
            self.offset = [0,0]
        else:
            self.offset = parent.offset[:]

        global counter
        self.id = counter
        counter = counter + 1

        self.sum = {'score':0.0, 'distinct':0, 'mode':[0,0], 'sum':[]}
        self.line_estimates = []
        self.features = []
        self.traces = []


        if offset is not None:
            self.offset[0] += offset[0]
            self.offset[1] += offset[1]

    def init_from_line_group(self, group):
        s0 = np.array([])
        s1 = np.array([])
        locs = []
        pslops = []
        slops = []
        for x in group:
            s0 = np.concatenate((s0,x['side-traces'][0]))
            s1 = np.concatenate((s1,x['side-traces'][1]))
            locs += x.side_locs
            slops.append(x.slope)
            pslops.append(x.pslope)

        p0 = group[0]['abs-line'][0]
        p1 = group[-1]['abs-line'][1]
        self.abs_line = (p0,p1)
        self.side_traces = (s0,s1)
        self.side_locs = locs
        self.colinear_group = group
        slop_common = stats.mode(slops)[0][0]
        indx = slops.index(slop_common)
        self.slope = slop_common
        self.pslope = pslops[indx]
        return self

    # backwards compatible with dict
    def __getitem__(self, key):
        return getattr(self,key.replace('-','_'))
    def __setitem__(self, key, value):
        return setattr(self,key.replace('-','_'),value)
    def __contains__(self,key):
        return hasattr(self,key)
    def __repr__(self,):
        return self.__str__()
    def __str__(self,):
        #return '%dx%d' % (self.width,self.height,)
        return '%dx%d' % (self.rect[0][0],self.rect[0][1])

def wrap_image(im,parent=None,offset=None):
    return Shape(im,parent,offset)


HORIZONTAL = 0
VERTICAL = 1
DIAGNOL = 2

class Dimension:
    ocr_group = None
    def __init__(self,tri1,tri2,):
        """
            Make a dimension object.  Triangles are each 3 points: (bp1, bp2, tip)
                Where the two base points are first and the tip is last
        """
        self.tri1 = tri1
        self.tri2 = tri2
        line = (tri1[2], tri2[2])
        m = line_slope(line)
        if m > CLAMP_SLOPE_UPPER:
            m = MAX_SLOPE
        elif m < CLAMP_SLOPE_LOWER:
            m = 0

        if m == MAX_SLOPE:
            self.type = VERTICAL
        elif m == 0:
            self.type = HORIZONTAL
        else:
            self.type = DIAGNOL

        self.line = line
        self.base_len = line_len(tri1[:2])

class TextBox:
    def __init__(self, pt1, pt2, text):
        self.width = abs(pt1[0] - pt2[0])
        self.height = abs(pt1[1] - pt2[1])
        self.p = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
        self.p1 = pt1
        self.p2 = pt2
        self.text = text



