import numpy as np
from scipy import stats
from rtree import index

class RTree():
    def __init__(self,shape,use_offset=True):
        self.features = {}
        self.tree = index.Index()
        self.shape = shape
        self.use_offset = use_offset

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

    def add_obj(self,x):
        left,bottom,right,top = self.get_bounds(x,)
        self.tree.insert(x['id'],(left,bottom,right,top))
        self.features[x['id']] = x
        #assert(x['id'] != 265)

    def add_objs(self,objs):
        for x in objs:
            self.add_obj(x)

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
            left += x['offset'][0]
            right += x['offset'][0]
            bottom += x['offset'][1]
            top += x['offset'][1]

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
            print('TREE is COHERENT with LIST')
        else:
            print(items)
            print(items2)
            print(len(items), 'vs', len(objs))


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
        self.group = group
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


def wrap_image(im,parent=None,offset=None):
    return Shape(im,parent,offset)
