import numpy as np

class Constraint:
    def __init__(self, num, soft=True):
        self.num = num
        self.soft = soft

class AutoConstrain:

    def rectangles(rectangles):
        for x in rectangles:
            # TR, BR, BL, TL, TR
            xdim = abs(x.rect[0][0] - x.rect[3][0])
            ydim = abs(x.rect[0][1] - x.rect[1][1])
            #x.

def infer_drawing(orig,ins):
    blank = (np.copy(orig) * 0) + 255
    AutoConstrain.rectangles(ins['rectangles'])

    return ins
