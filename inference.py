import numpy as np

from utils import save
from cli import put_thing
from slvs import *

class Constrainer:
    def __init__(self,):
        self.sys = Slvs()

    def Point2D(self,p1):
        return Point2D(self.sys,p1)

    def Line(self,p1,p2):
        return Line(self.sys,p1,p2)

    def solve(self,):
        return self.sys.solve()

    def constr(self,op,val,p1,p2,e1,e2):
        return self.sys.MakeConstraint(op, val, p1, p2, e1, e2);

    def add_rectangle(self,r):
        TR = self.Point2D(r[0])
        BR = self.Point2D(r[1])
        BL = self.Point2D(r[2])
        TL = self.Point2D(r[3])

        top = self.Line(TL,TR)
        bottom = self.Line(BL,BR)
        left = self.Line(TL,BL)
        right = self.Line(BR,TR)

        self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, top, bottom);
        self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, left, right);
        self.constr(SLVS_C_PERPENDICULAR, 0, 0, 0, top, right);
        self.constr(SLVS_C_HORIZONTAL, 0, 0, 0, top, 0);

        return (top,bottom,left,right)


class Point2D:
    def __init__(self,sys,p1):
        self.x = sys.P(p1[0])
        self.y = sys.P(p1[1])
        self.obj = sys.MakePoint2d(self.x, self.y);
    def __int__(self,):
        return self.obj
    def __getitem__(self,i):
        return self.p[i]

    def array(self):
        return [self.x.val,self.y.val]

class Line:
    def __init__(self,sys,p1,p2):
        if isinstance(p1, Point2D):
            self.p1 = p1
        else:
            self.p1 = Point2D(sys,p1)
        if isinstance(p2, Point2D):
            self.p2 = p2
        else:
            self.p2 = Point2D(sys,p2)
        self.obj = sys.MakeLineSegment(self.p1, self.p2);
    def __int__(self,):
        return self.obj
    def __getitem__(self,i):
        return self.line[i]
    def array(self):
        return [self.p1.array(),self.p2.array()]

#class Rectangle


class AutoConstrain:

    def rectangles(orig,rectangles):
        sys = Constrainer()
        geo = []

        for x in rectangles:
            # TR, BR, BL, TL, TR
            #xsz = abs(x.rect[0][0] - x.rect[3][0])
            #ysz = abs(x.rect[0][1] - x.rect[1][1])
            lines = sys.add_rectangle(x.rect)

            geo.append(lines)

        res = sys.solve()
        assert(res.status == SLVS_RESULT_OKAY)

        print('dof:', res.dof)
        for lset in geo:
            for i in lset:
                put_thing(orig,i.array(),(255,0,0))



            

def infer_drawing(orig,ins):
    blank = (np.copy(orig) * 0) + 255
    AutoConstrain.rectangles(blank,ins['rectangles'])

    save(blank,'output2.png')
    return ins
