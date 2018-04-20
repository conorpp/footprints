import numpy as np

from utils import save,line_len
from cli import put_thing,Context
from slvs import *

from preprocessing import group_rects
from context_correction import infer_ocr_groups

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

        self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, top, bottom)
        self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, left, right)
        self.constr(SLVS_C_PERPENDICULAR, 0, 0, 0, top, right)
        self.constr(SLVS_C_HORIZONTAL, 0, 0, 0, top, 0)

        return (top,bottom,left,right)

    def same_rects(self,r1,r2, rotate=False):
        """ Constraint two rectangles to have same height and width """
        if rotate:
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[0], r2[2])
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[2], r2[0])
        else:
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[0], r2[0])
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, r1[2], r2[2])

    def colinear_rects(self,r1,r2, horizontal,orig=None):
        """ Constraint two same-sized rectangles to have colinear sides """

        if horizontal:
            r1p1 = r1[0].p1
            r1p2 = r1[0].p2
            r2p1 = r2[0].p1
            r2p2 = r2[0].p2
        else:
            r1p1 = r1[2].p1
            r1p2 = r1[2].p2
            r2p1 = r2[2].p1
            r2p2 = r2[2].p2
        
        if line_len((r1p1.array(),r2p2.array())) > line_len((r1p2.array(),r2p1.array())):
            p1 = r1p1
            p2 = r2p2
            pn = r1p2
        else:
            p1 = r1p2
            p2 = r2p1
            pn = r1p1

        colin = self.Line(p1,p2)

        if orig is not None:
            l = [(p1.x.val,p1.y.val),(p2.x.val,p2.y.val)]
            l2 = [(pn.x.val-2,pn.y.val-2),(pn.x.val+2,pn.y.val+2)]
            put_thing(orig,l,(255,0,0),None,3)
            put_thing(orig,l2,(0,200,0),None,3)

        if horizontal:
            self.constr(SLVS_C_HORIZONTAL, 0, 0, 0, colin, 0)
        else:
            self.constr(SLVS_C_VERTICAL, 0, 0, 0, colin, 0)


    def same_pitch_rects(self,rects):
        lines = []
        for i,r1 in enumerate(rects[:len(rects)-1]):
            r2 = rects[i+1]
            p1 = r1[0].p1
            p2 = r2[0].p1
            l = self.Line(p1,p2)
            lines.append(l)

        for i,l1 in enumerate(lines[:len(lines)-1]):
            l2 = lines[i+1]
            self.constr(SLVS_C_EQUAL_LENGTH_LINES, 0, 0, 0, l1, l2)
        #print('%d SAME LINES' % (len(lines)-1))




class Point2D:
    def __init__(self,sys,p1):
        self.x = sys.P(p1[0])
        self.y = sys.P(p1[1])
        self.obj = sys.MakePoint2d(self.x, self.y);
    def __int__(self,):
        return self.obj
    def __getitem__(self,i):
        if i == 0: return self.x
        elif i==1: return self.y
        raise IndexError('Index %d out of range, only use (0,1)' % i)

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

def group_data(data, lam):
    groups = []
    for x in data:
        added = False
        for g in groups:
            if lam(x,g[0]):
                g.append(x)
                added = True
                break
        if not added:
            groups.append([x])
    return groups



def organize_by_pitch(group,axis):
    if len(group) < 2: return []
    arr = np.array([x.rect[0][axis] for x in group])
    diff = np.diff(arr)
    for i,r in enumerate(group):
        if i == 0:
            r.diff = diff[i]
        elif i < len(diff):
            r.diff = min(diff[i],diff[i-1])
        else:
            r.diff = diff[i-1]

    def close_pitch(x,y):
        return x.diff < 4

    return group_data(group, close_pitch)



def organize_by_alignment(group):
    def by_xpitch(x,y):
        x = x.rect[0][0]
        y = y.rect[0][0]
        return x < (y + 5) and x > (y - 5)
    def by_ypitch(x,y):
        x = x.rect[0][1]
        y = y.rect[0][1]
        return x < (y + 5) and x > (y - 5)

    xgroups = group_data(group, by_xpitch)
    ygroups = group_data(group, by_ypitch)

    return xgroups,ygroups

def get_bounding_rect(rects):
    xmin = rects[0].rect[0][0]
    xmax = rects[0].rect[0][0]
    ymin = rects[0].rect[0][1]
    ymax = rects[0].rect[0][1]
    
    for r in rects:
        for p in r.rect:
            x,y = p
            xmin = min(xmin,x)
            xmax = max(xmax,x)
            ymin = min(ymin,y)
            ymax = max(ymax,y)

    return (xmin,xmax,ymin,ymax)


class Output:
    def draw_bounding_rect(orig,rects):

        (xmin,xmax,ymin,ymax) = get_bounding_rect(rects)

        xmin -= 1
        ymin -= 1
        ymax += 1
        xmax += 1

        r = [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)]

        put_thing(orig,r,(0,0,255))





class AutoConstrain:

    def rectangles(sys, rectangles, orig = None):

        for x in rectangles:
            lines = sys.add_rectangle(x.rect)
            x.geom = lines

        groups = group_rects(rectangles,12)

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

        for i,g in enumerate(groups):
            g = g[2]
            #print('group %d' % i)
            xsubs,ysubs = organize_by_alignment(g)
            if len(xsubs) < len(ysubs):
                horizontal = 0
                subgroups = xsubs
            else:
                horizontal = 1
                subgroups = ysubs
            axis = (horizontal + 1) & 1
            #print('  %d subbgroups' % (len(subgroups)))

            subgroup_stats = []
            for k, sg in enumerate(subgroups):
                if orig is not None: Output.draw_bounding_rect(orig,sg)
                for j, r1 in enumerate(sg[:len(sg)-1]):
                    r2 = sg[j+1]
                    sys.colinear_rects(r1.geom,r2.geom,horizontal)

                pitches = organize_by_pitch(sg, axis)
                for ssg in pitches:
                    #print('  %d rects in pitch group' % (len(ssg)))
                    sys.same_pitch_rects([r.geom for r in ssg])
                
                subgroup_stats.append(
                        {
                            'pitch_groups':pitches,
                            'bound_rect': get_bounding_rect(sg),
                            'subgroup': sg
                        })
                #print('  subgroup %d has %d pitch groups' % (k, len(pitches)))

            def vertically_aligned(x,y):
                xbound = x['bound_rect']
                ybound = y['bound_rect']
                dmin1 = xbound[axis*2 + 0]  # axis is delegate
                dmax1 = xbound[axis*2 + 1]
                dmin2 = ybound[axis*2 + 0]
                dmax2 = ybound[axis*2 + 1]
                same_width = within_margin(dmin1,dmin2,4) and within_margin(dmax1,dmax2,4)
                same_pitch = len(x['pitch_groups']) == len(y['pitch_groups'])
                return same_width and same_pitch

            valigned = group_data(subgroup_stats, vertically_aligned)
            #print('  %d sets of subgroups aligned to each other' % (len(valigned)))
            for sg in valigned:
                #sg = sg['subgroup']
                for i,ssg1 in enumerate(sg[:len(sg)-1]):
                    ssg2 = sg[i+1]
                    r1 = ssg1['subgroup'][0]
                    r2 = ssg2['subgroup'][0]
                    sys.colinear_rects(r1.geom,r2.geom, axis)

                #print('    %d groups aligned' % (len(sg)))


        #r1 = groups[1][2][0]
        #top1 = groups[2][2][0].geom[0]
        #top2 = groups[2][2][7].geom[0]
        #r1.geom[0].p1.y.val -= 50
        #xdist = abs(top.p1.x.val - top.p2.x.val)-25
        #sys.constr(SLVS_C_PT_PT_DISTANCE, 700, top1.p1, top2.p1, 0, 0);

        #for i,g in enumerate(groups):
            #g = g[2]
            #horz,verz = infer_ocr_groups(orig,g,5000)
            #print('group %d has %d horz, %d verz' %(i, len(horz), len(verz)))
            #Context.draw_ocr_group_rects(orig, horz,verz)



            

def infer_drawing(orig,ins):
    blank = (np.copy(orig) * 0) + 255
    sys = Constrainer()
    AutoConstrain.rectangles(sys, ins['rectangles'], blank)

    res = sys.solve()
    assert(res.status == SLVS_RESULT_OKAY)

    print('dof:', res.dof)
    for r in ins['rectangles']:
        lset = r.geom
        for i in lset:
            put_thing(blank,i.array(),(255,0,0))


    save(blank,'output2.png')
    return ins
