import numpy as np

from utils import save
from cli import put_thing
from slvs import *

from preprocessing import group_rects

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
        """ Input corners of rectangle and add rectangle constraints for them.  
            Return parameters. """
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

    def same_rects(self,r1,r2, rotate=False):
        """ Constraint two rectangles to have same height and width """
        if rotate:
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[0], r2[2]);
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[2], r2[0]);
        else:
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[0], r2[0]);
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[2], r2[2]);



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
            lines = sys.add_rectangle(x.rect)

            # (top,bottom,left,right)
            geo.append(lines)
            x.geom = lines

        groups = group_rects(rectangles,12)
        print('there are %d groups rectangles' % len(groups))
        for g in groups:
            g = g[2]
            for i,r1 in enumerate(g[:len(g)-1]):
                r2 = g[i+1]
                sys.same_rects(r1.geom,r2.geom)

        def within_margin(v1,v2,mar):
            return v1 < (v2+mar) and v1 > (v2-mar)

        for i,g1 in enumerate(groups):
            for j,g2 in enumerate(groups):
                if g1 is g2: continue
                xdim1,ydim1,_ = g1
                xdim2,ydim2,_ = g2
                if xdim1 == 0: continue
                if within_margin(xdim1,ydim2,12) and within_margin(xdim2,ydim1,12):
                    print('group %d is same as %d' % (i,j))
                    sys.same_rects(g1[2][0].geom, g2[2][0].geom, True)
                    g2[0] = 0
                    g2[1] = 0


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
