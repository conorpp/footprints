from rtree import index

class RTree():
    def __init__(self,arr):
        self.features = {}
        self.tree = index.Index()
        self.shape = arr.shape

    def has(self,idx):
        return (idx in self.features)

    def get_bounds(self, x):
        left,bottom = x['boundxy']
        right,top = left + x['width'], bottom+ x['height']
        left += x['offset'][0]
        right += x['offset'][0]
        bottom += x['offset'][1]
        top += x['offset'][1]
        return left,bottom,right,top

    def add_obj(self,x):
        left,bottom,right,top = self.get_bounds(x)
        self.tree.insert(x['id'],(left,bottom,right,top))
        self.features[x['id']] = x
        #assert(x['id'] != 265)
        if x['id'] == 265:
            print('ADDED 265')

    def add_objs(self,objs):
        for x in objs:
            self.add_obj(x)

    def remove(self,idx):
        #if bound is None:
        bound = self.get_bounds(self.features[idx])
        self.tree.delete(idx,bound)
        del self.features[idx]

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

    def __init__(self,im,parent=None, offset=None):
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

    # backwards compatible with dict
    def __getitem__(self, key):
        return getattr(self,key.replace('-','_'))
    def __setitem__(self, key, value):
        return setattr(self,key.replace('-','_'),value)
    def __contains__(self,key):
        return hasattr(self,key)


def wrap_image(im,parent=None,offset=None):
    return Shape(im,parent,offset)
