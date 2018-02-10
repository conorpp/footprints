from rtree import index

class RTree():
    def __init__(self,arr):
        self.features = {}
        self.tree = index.Index()
        self.shape = arr.shape

    def has(self,idx):
        return (idx in self.features)

    def add_obj(self,x):
        left,bottom = x['boundxy']
        right,top = left + x['width'], bottom+ x['height']
        left += x['offset'][0]
        right += x['offset'][0]
        bottom += x['offset'][1]
        top += x['offset'][1]
        self.tree.insert(x['id'],(left,bottom,right,top))
        self.features[x['id']] = x

    def add_objs(self,objs):
        for x in objs:
            self.add_obj(x)

    def remove(self,idx, bound = None):
        if bound is None:
            bound = self.shape
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


