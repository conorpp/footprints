import math
from random import randint

from scipy import stats
from utils import *
from filters import *
from analyzers import *
from processors import *
from plotting import plotfuncs, dump_plotly
from dimension_correction import TriangleHumps
from cli import put_thing
from structures import RTree
import preprocessing

def group_ocr_lines(ocr,dim=0):
    """  """
    odim = (dim+1)&1

    indx = 'boundxy3' if dim else 'boundxy2'

    ocr = sorted(ocr, key = lambda x : x[indx][odim])

    ocr_groups = []
    group = []
    starty = ocr[0][indx][odim]
    # separate into groups close on dimension
    for x in ocr:
        y = x[indx][odim] 

        if y > (starty-5) and y < (starty+5):
            group.append(x)
        else:
            if len(group) > 0:
                ocr_groups.append(group)
            group = [x]
            starty = y
    if len(group) > 0:
        ocr_groups.append(group)

    ocr_groups = [sorted(x, key = lambda i : i[indx][dim]) for x in ocr_groups]
    return ocr_groups

def wrap_image_ocr(image,parent, char, conf, tl, h, w):
    """ super to wrap_image """
    image = wrap_image(image,parent)
    image['symbol'] = char
    image['ocr-conf'] = conf
    image['boundxy'] = tl
    image['height'] = h
    image['width'] = w
    return image

def detect_punct(arr, x1, x2, dx, dim, ):

    """ Detect black pixels in the 'punctuation-region' and call it a period. """

    odim = (dim+1)&1
    indx = 'boundxy3' if dim else 'boundxy2'
    indx2 = 'height' if dim else 'width'
    oindx2 = 'width' if dim else 'height'

    # points of the rectangle between characters
    bl = (x1[indx][dim] + x1[indx2], x1[indx][odim])
    tl = (bl[0], bl[1] - x1[oindx2])
    tr = (tl[0] + abs(dx), tl[1])
    br = (tr[0], bl[1])

    dy1 = abs(bl[1] - tr[1])
    dy2 = abs(br[1] - tl[1])
    if dy2>dy1:
        y1 = tl[1]
        y2 = br[1]
        dy = dy2
    else:
        y1 = tr[1]
        y2 = bl[1]
        dy = dy1
    dy = max(dy1,dy2)
    #print('    %dx%d' % (dx,dy))

    #print('potential punct %d->%d, %d->%d' % (y1,y2+1, bl[0] , br[0]+1))
    if dim == 0:
        #print('potential punct %d->%d, %d->%d' % (y1,y2+1, bl[0] , br[0]+1))
        inbetw = arr['img'][y1:y2+1, bl[0] : br[0]+1]
        low_third = inbetw[int(inbetw.shape[0]*2/3):inbetw.shape[0], :]
        rest_third = inbetw[:int(inbetw.shape[0]*2/3), :]
        if count_black(low_third) > (4*count_black(rest_third)):
            newim = wrap_image_ocr(inbetw,arr, '.',.90, tl, dx, dy)
            return newim
    else:
        #print('potential punct %d->%d, %d->%d' % (bl[0] , br[0]+1,y1,y2+1))
        inbetw = arr['img'][bl[0] : br[0]+1,y1:y2+1]
        low_third = inbetw[:,int(inbetw.shape[1]*2/3):inbetw.shape[1]]
        rest_third = inbetw[:,:int(inbetw.shape[0]*2/3)]
        if count_black(low_third) > (4*count_black(rest_third)):
            newim = wrap_image_ocr(inbetw,arr, '.',.90, tr, dy, dx)
            return newim
    return None


def group_ocr(arr, ocr, dim=0):
    """ Naively group adjacent OCR characters in same dimension row/col. Good first step. """
    # dim=0 means horizontal text, dim=1 means vertical text
    odim = (dim+1)&1

    indx = 'boundxy3' if dim else 'boundxy2'
    indx2 = 'height' if dim else 'width'
    oindx2 = 'width' if dim else 'height'

    # separate into groups close on dimension
    ocr_groups = group_ocr_lines(ocr,dim)

    ocr_groups2 = []
    SEPARATING_DIST = 4         # TODO normalize
    SEPARATING_DIST_COMMA = 36  # TODO normalize

    # Segment into consecutive groups on other dimension
    for i,group in enumerate(ocr_groups):
        x1 = group[0]
        segment = [x1]

        for x2 in group[1:]:
            dx = (x2[indx][dim] - x1[indx][dim] - x1[indx2])

            if abs(dx) > SEPARATING_DIST:

                punct = None
                if dx < SEPARATING_DIST_COMMA:
                    punct = detect_punct(arr, x1, x2, dx, dim)

                if punct is not None:
                    #print(x1['symbol'],'->', x2['symbol'])
                    #print(punct['img'])
                    #print('itsa punct')
                    segment.append(punct) # detected comma/decimal
                    segment.append(x2)
                else:
                    if len(segment) > 0:
                        ocr_groups2.append(segment)
                    segment = [x2]
            else:
                segment.append(x2)

            x1 = x2
        
        if len(segment) > 0:
            ocr_groups2.append(segment)

    ocr_groups = ocr_groups2
    widths = []
    heights = []
    for group in ocr_groups:
        for x in group:
            widths.append(x['width'])
            heights.append(x['height'])

    #print('widths', widths)
    if len(widths):
        wmode = stats.mode(widths)[0][0]
        hmode = stats.mode(heights)[0][0]
    else:
        wmode = 0
        hmode = 0
    #print(mode)

    ocr_groups2 = []
    segment = []

    return ocr_groups


def group_crosses(group1,group2):
    """ support.  Test if two OCR groups cross each other. """
    for x in group1:
        for y in group2:
            if x['id'] == y['id']:
                return True
    return False
def item_contained(x,group_set):
    """ support.  Test if a OCR group is contained by a set of OCR groups. """
    for group2 in group_set:
        for y in group2:
            if x['id'] == y['id']:
                return True
    return False



def block_redundant_groups(horz, verz):
    """ block groups contained in other groups.  E.g. A verticle group contained by multiple horizontal groups. """
    new_horz = []
    new_verz = []
    
    for group in horz:
        if len(group)>1:
            new_horz.append(group)
 
    for group in verz:
        if len(group)>1:
            new_verz.append(group)
    #print(len(verz+horz),'initial groups')

    crossed = False
    for group1 in horz:
        if len(group1) == 1:
            for group2 in new_verz:
                if group_crosses(group1,group2):
                    # remove group1
                    crossed = True
                    break
            
            if not crossed:
                new_horz.append(group1)
            crossed = False


    for group1 in verz:
        if len(group1) == 1:
            for group2 in new_horz:
                if group_crosses(group1,group2):
                    # remove group1
                    crossed = True
                    break
            
            if not crossed:
                new_verz.append(group1)
            crossed = False

    # block redundant vertical text.
    verz = []
    for group1 in new_verz:
        items = len(group1)

        for x in group1:
            if x['symbol'] == '.' or item_contained(x,new_horz):
                items -= 1
            else:
                break

        if items != 0:
            verz.append(group1)
    new_verz = verz

    # block redundant horizontal text.
    horz = []
    for group1 in new_horz:
        items = len(group1)

        for x in group1:
            if x['symbol'] == '.' or item_contained(x,new_verz):
                items -= 1
            else:
                break

        if items != 0:
            horz.append(group1)
    new_horz = horz



    return new_horz,new_verz


def infer_ocr_groups(arr, ocr):
    """ Group together OCR characters.  Ensure rotation coherency. """
    widths = []
    heights = []
    for x in ocr:
        widths.append(x['width'])
        heights.append(x['height'])

    wmode = stats.mode(widths)[0][0]
    hmode = stats.mode(heights)[0][0]
    mode = max(wmode,hmode,35)

    #print(wmode)
    #print(hmode)

    ocr = [x for x in ocr if x['width'] < (mode*2)]
    ocr = [x for x in ocr if x['height'] < (mode*2)]

    # get starting x,y for bottom left coord, with added offset
    for x in ocr:
        x['boundxy2'] = (x['offset'][0] + x['boundxy'][0], x['offset'][1] + x['boundxy'][1] + x['height'])
        x['boundxy3'] = (x['offset'][0] + x['boundxy'][0] + x['width'], x['offset'][1] + x['boundxy'][1])

    #vert,horiz = if_rotated(ocr)
    #print('horz:')
    horz = group_ocr(arr,ocr, 0)
    #print('verz:')
    verz = group_ocr(arr,ocr, 1)

    new_horz,new_verz = block_redundant_groups(horz,verz)

    # correct for rotations
    not_rotated = []
    for group in new_verz:
        for x in group:
            if not x['rotated'] and x['symbol'] != '.':
                oldsym = x['symbol']
                rotate_right([x])
                analyze_ocr([x])
                rotate_left(not_rotated)
                if x['symbol'] is None:
                    x['symbol'] = oldsym
                    x['rotated'] = False

    return new_horz, new_verz

def draw_ocr_group_rects(orig, new_horz, new_verz):
    """ output function for drawing rectangles around the OCR groups """
    print(len(new_horz) + len(new_verz),'groups')
    for i,group in enumerate(new_horz):
        leftest = group[0]
        rightest = group[-1]

        mytop = min(leftest['boundxy2'][1] - leftest['height'],rightest['boundxy2'][1] - rightest['height'] )
        mybot = max(leftest['boundxy2'][1],rightest['boundxy2'][1])

        pt1 = (leftest['boundxy2'][0], mytop)
        pt2 = (rightest['boundxy2'][0] + rightest['width'], mybot)


        cv2.rectangle(orig, pt1,pt2, [0,0,255])
        #s = ''.join(x['symbol'] for x in group)
        #print(s)
        #print(pt1, pt2)
        #if i == 9:
            #break
    for i,group in enumerate(new_verz):
        leftest = group[0]
        rightest = group[-1]

        mytop = min(leftest['boundxy2'][1] - leftest['height'],rightest['boundxy2'][1] - rightest['height'] )
        mybot = max(leftest['boundxy2'][1],rightest['boundxy2'][1])

        pt1 = (leftest['boundxy2'][0], mytop)
        pt2 = (rightest['boundxy2'][0] + rightest['width'], mybot)

        cv2.rectangle(orig, pt1,pt2, [0,180,0])
        #s = ''.join(x['symbol'] for x in group)[::-1]
        #print(s)
        #if i == 4:
            #break




def untangle_circles(circles, ocr_groups):
    """ Remove circles from OCR characters like P,8,O,... """

    new_circles = []
    rejects = []
    for c in circles:
        center,r = c['circle']
        r -= 1
        coff = c['offset']
        ctl = (center[0] - r + coff[0], center[1] - r + coff[1])
        ctr = (center[0] + r + coff[0], center[1] - r + coff[1])
        cbl = (center[0] - r + coff[0], center[1] + r + coff[1])
        cbr = (center[0] + r + coff[0], center[1] + r + coff[1])
        add = True
        for group in ocr_groups:
            leftest = group[0]
            rightest = group[-1]
            
            mytop = min(leftest['boundxy2'][1] - leftest['height'],rightest['boundxy2'][1] - rightest['height'] )
            mybot = max(leftest['boundxy2'][1],rightest['boundxy2'][1])

            rtl = (leftest['boundxy2'][0], mytop)
            rbr = (rightest['boundxy2'][0] + rightest['width'], mybot)
            rtr = (rbr[0], rtl[1])
            rbl = (rtl[0], rbr[1])
            remove_group = False
            if ctl[0] >= rtl[0] and ctl[1] >= rtl[1]:
                if ctr[0] <= rtr[0] and ctr[1] >= rtr[1]:
                    if cbr[0] <= rbr[0] and cbr[1] <= rbr[1]:
                        if cbl[0] >= rbl[0] and cbl[1] <= rbl[1]:
                            if len(group)>1:
                                add = False
                                break
                            elif group[0]['symbol'] in '689':
                                add = False
                                break
                            elif group[0]['symbol'] in '0QC':
                                remove_group = True
                                break
                            #s = ''.join(x['symbol'] for x in group)
                            #print('circle contained by group '+s)
        if remove_group:
            rejects.append(group)

        if add:
            new_circles.append(c)
        else:
            break
        add = True
    return new_circles, rejects

def remove_ocr_groups(ocr_groups, ocr_rejects=None):
    """ support function to untangle circles from OCR groups """
    if ocr_rejects is not None:
        for group in ocr_rejects:
            for x in group:
                x['symbol'] = None
                x['ocr-conf'] = 0.0

    new_ocrs = []
    for group in ocr_groups:
        if group[0]['symbol'] != None:
            new_ocrs.append(group)

    return new_ocrs

def combine_features(arr, t1,t2, pts=None):
    """ Combine images and draw line between them """
    t1off = t1['offset']
    t2off = t2['offset']
    bg = np.zeros(arr.shape,dtype=np.uint8)+255

    bg[t1off[1]:t1['img'].shape[0] + t1off[1], t1off[0]:t1['img'].shape[1] + t1off[0]] = t1['img']
    bg[t2off[1]:t2['img'].shape[0] + t2off[1], t2off[0]:t2['img'].shape[1] + t2off[0]] = t2['img']

    if pts is None:
        p1 = centroid(t1['ocontour'])
        p2 = centroid(t2['ocontour'])
        p1 = (p1[0] + t1off[0], p1[1] + t1off[1])
        p2 = (p2[0] + t2off[0], p2[1] + t2off[1])
    else:
        p1,p2 = pts

    cv2.line(bg, p1, p2, 0, 2)

    bg[0:arr.shape[0],0:2] = 255
    bg[0:arr.shape[0],arr.shape[1]-2:arr.shape[1]] = 255
    bg[0:2,0:arr.shape[1]] = 255
    bg[arr.shape[0]-2:arr.shape[0],0:arr.shape[1]] = 255



    newim = wrap_image(bg, t1)
    newim['offset'] = np.array((0,0))
    return newim



def coalesce_triangles(arr,triangles, tree):
    """ merge triangles adjacent to each other """
    def find_merges(tri_set):  #this could time a long time with many triangles
        merges = []
        for x in tri_set:
            s1off = x[0]['offset']
            tri1,s1 = x
            for y in tri_set:
                if x is y:
                    continue
                if y[0]['marked']:
                    continue

                s2off = y[0]['offset']
                tri2,s2 = y


                l1 = min(line_len((s1[0] + s1off, s2[0] + s2off)), line_len((s1[0] + s1off, s2[1] + s2off)))
                l2 = min(line_len((s1[1] + s1off, s2[0] + s2off)), line_len((s1[1] + s1off, s2[1] + s2off)))

                lmin = min(l1,l2)

                if lmin<6: # TODO centralize this, based on line thickness basically
                    l1m = max(line_len((s1[0] + s1off, s2[0] + s2off)), line_len((s1[0] + s1off, s2[1] + s2off)))
                    l2m = max(line_len((s1[1] + s1off, s2[0] + s2off)), line_len((s1[1] + s1off, s2[1] + s2off)))
                    lmax = max(l1m,l2m)

                    ssum = (line_len(s1) + line_len(s2))
                    #print('lextrema %.1f %.1f' % (lmax, lmin))
                    #print('side sum', ssum)

                    if lmax < ssum:
                        #merges.append((x[0],y[0],lmin,lmax))
                        merges.append((x[0],y[0]))

            x[0]['marked'] = True
        return merges

    merged = []
    merges = []
    for x in triangles:
        nearby = tree.intersect(x,1)
        #nearby.append(x)
        has_hor = []
        has_ver = []

        for x in nearby:
            if x['merged']: continue
            t = 0
            side1 = (x['triangle'][0], x['triangle'][1])
            side2 = (x['triangle'][1], x['triangle'][2])
            side3 = (x['triangle'][0], x['triangle'][2])
            x['marked'] = False
            for side in (side1,side2,side3):
                #print(side)
                vert = line_vert(side)
                horz = line_horz(side)
                if vert:
                    has_ver.append((x,side))
                    break
                if horz:
                    has_hor.append((x,side))
                    break
        #print('tris',len(has_hor), len(has_ver))

        m = find_merges(has_hor) + find_merges(has_ver)
        for x in m:
            x[0]['merged'] = True
        merges += m

    merges2 = []
    id_list = []
    for t1,t2 in merges:
        if t1['id'] in id_list or t2['id'] in id_list:
            continue
        id_list.append(t1['id'])
        id_list.append(t2['id'])
        merges2.append((t1,t2))
    merges = merges2
    # merge the merges
    for t1,t2 in merges:
        newim = combine_features(arr,t1,t2)
        analyze_rectangles((newim,))
        analyze_triangles((newim,),arr)
        merged.append(newim)
        tree.remove(t1['id'])
        tree.remove(t2['id'])
        tree.add_obj(newim)



    new_triangles = merged[:]
    for x in triangles:
        is_merged = False
        for t1,t2 in merges:
            if (x is t1) or (x is t2):
                is_merged = True
                break
            #else:
                #new_triangles.append(x)
                #break
        if not is_merged:
            new_triangles.append(x)
            #break
    if len(merged) == 0:
        new_triangles = triangles


    return new_triangles,merges,merged

class Updates:
    """
        Stuff that should have been done correctly before,
        but now just going to compensate for it here.
    """
    def update(outs):
        Updates.circles(outs['circles'])
        Updates.triangles(outs['triangles'])
        Updates.lines(outs['lines'])
        Updates.rectangles(outs['rectangles'])

    def circles(circles):
        for x in circles:
            center,r = x['circle']
            xmin = center[0]-r
            ymin = center[1]-r
            x['boundxy'] = (xmin,ymin)
            x['width'] = 2 * r
            x['height'] = 2 * r
            x['offset'] = np.array(x['offset'])
            #x['offset'] = np.array(x['offset'])
            #x['offset'] = np.reshape(np.array(x['offset']),(2))

#def update_bounding_boxes(outs):
    #""" update bounding boxes to be absolute and tight bound"""
    def triangles(triangles):
        for x in triangles:
            x['triangle'] = np.reshape(x['triangle'], (3,2))
            xmin = np.min(x['triangle'][:,0])
            xmax = np.max(x['triangle'][:,0])
            ymin = np.min(x['triangle'][:,1])
            ymax = np.max(x['triangle'][:,1])
            x['boundxy'] = (xmin,ymin)
            x['width'] = xmax-xmin
            x['height'] = ymax-ymin
            x['offset'] = np.array(x['offset'])
            #x['triangle'] = np.reshape(x['triangle'],(3,2))
            #x['offset'] = np.array(x['offset'])
            #x['offset'] = np.reshape(np.array(x['offset']),(2))

    def lines(lines):
        for x in lines:
            xmin = min(x['abs-line'][0][0], x['abs-line'][1][0],)
            xmax = max(x['abs-line'][0][0], x['abs-line'][1][0],)
            ymin = min(x['abs-line'][0][1], x['abs-line'][1][1],)
            ymax = max(x['abs-line'][0][1], x['abs-line'][1][1],)

            if 'ocontour' not in x:
                analyze_rectangles((x,))
            x['boundxy'] = (xmin,ymin)
            x['width'] = xmax-xmin
            x['height'] = ymax-ymin
            x['offset'] = np.array(x['offset'])
            #x['offset'] = np.reshape(np.array(x['offset']),(2))

    def rectangles(rectangles):
        for x in rectangles:
            xmin = np.min(x['contour'][:,0])
            xmax = np.max(x['contour'][:,0])
            ymin = np.min(x['contour'][:,1])
            ymax = np.max(x['contour'][:,1])
            x['boundxy'] = (xmin,ymin)
            x['width'] = xmax-xmin
            x['height'] = ymax-ymin
            x['offset'] = np.array(x['offset'])
            #x['offset'] = np.reshape(np.array(x['offset']),(2))
            #print(x['offset'])

    #for x in outs['irregs']:
        #pass # TODO

def add_abs_line_detail(lines):
    """ calculate more information for lines """
    nlines = []
    for x in lines:
        l = x['line']
        p1 = l[0] + x['offset']
        p2 = l[1] + x['offset']
        d1 = line_len(((0,0), p1))
        d2 = line_len(((0,0), p2))
        if d1 > d2:
            x['abs-line'] = np.array((p2,p1))
        else:
            x['abs-line'] = np.array((p1,p2))

        x['slope'] = line_slope(x['abs-line'])


def line_intersection(L1, L2):
    """ return point where two lines intersect, false if they don't """
    def linecoeffs(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    L1 = linecoeffs(L1[0], L1[1])
    L2 = linecoeffs(L2[0], L2[1])

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

# 
def does_intersect(l0,l1):
    """ Return true if line segments l0 and l1 intersect """
    A,B,C,D = l0[0], l0[1], l1[0], l1[1]
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def angle_between_lines(l0,l1):
    """ Return angle between lines """
    ang1 = math.atan2(l0[0][1] - l0[1][1],
                      l0[0][0] - l0[1][0],)
    ang2 = math.atan2(l1[0][1] - l1[1][1],
                      l1[0][0] - l1[1][0],)
    diff = int(round((ang2 - ang1)* 180/math.pi))

    if diff < 0:
        diff += 360
    if diff > 270:
        diff -= 270
    if diff > 180:
        diff -= 180
    if diff > 90:
        diff = 180 - diff

    #print(diff)
    #if diff > 90:
        #diff = 90 - (diff%90)
    #if diff < 0:
        #diff += 360
    #diff -= 90 * int((diff-1)/90)
    return diff 

def correct_trimmed_triangles(triangles, line_tree):
    """ [not used] merge triangles with lines that cut into triangle """
    affected = []
    total_lines = 0
    for i,x in enumerate(triangles):
        #if i not in range(3,5):continue
        nearby_lines = line_tree.intersect(x,)
        tri = x['triangle']
        side0 = tri[0:2] #+ tri_off
        side1 = tri[1:3] #+  tri_off
        side2 = (tri[0],tri[2]) #+ tri_off
        shortest = min((side0,side1,side2), key = lambda x: line_len(x))
        maxdim = max(x['height'], x['width'])

        line_set = []
        for l in nearby_lines:
            doff = l['offset'] - x['offset']
            p0 = tuple(l['line'][0]+doff)
            p1 = tuple(l['line'][1]+doff)
            p0in = point_ineq_contour(tri, p0)
            p1in = point_ineq_contour(tri, p1)
            if line_len(l['line']) < maxdim * 2:
                if p0in and p1in:
                    affected.append((x,(l, None,)))
                    line_set = []
                    break
                elif p0in or p1in:
                    line_set.append((l,p0 if p0in else p1))
                    total_lines += 1

        line_set2 = []
        tri_off = x['offset']

        for l in line_set:
            good = False
            crossing_side = None
            not_intersects = 0
            for s in (side0,side1,side2):
                if s is shortest: continue
                doff = l[0]['offset'] - tri_off
                line = l[0]['line'] + doff
                if does_intersect(line,s):
                    ang = angle_between_lines(line,s)
                    if ang < 70 and ang > 20 :  # check easy bounds
                        good = True
                        crossing_side = s
                        #print('good one %d ' % (ang,))
                        break
                    elif ang <89 and ang > 1:  # take more extreme bounds only if line is inside triangle a good amount
                        inter_pt = line_intersection(line,s)
                        ln1 = line_len(line)
                        ln2 = line_len((inter_pt,l[1]))
                        if ln2/ln1 > .15:
                            crossing_side = s
                            good = True
                            #print('good one %d @%.2f' % (ang,ln2/ln1))
                            break
                    #else:
                        #print('bad one %d ' % (ang,))
                else:
                    #print('does not intersect')
                    not_intersects += 1
            if not_intersects == 3:
                print('Line did not intersect!')
                p0,p1= (l[0]['line'] + doff)
                p0,p1=tuple(p0),tuple(p1)
                p0in = point_ineq_contour(tri, p0)
                p1in = point_ineq_contour(tri, p1)
                assert(p0in or p1in)

            if good:
                line_set2.append((l[0],crossing_side + tri_off,ang))
        
        if len(line_set2) == 0:
            line_pick = False
            pass
        elif len(line_set2) == 1:
            line_pick = line_set2[0]
            affected.append((x,line_pick))
            #if line_pick is not False:
                #if line_pick[1] is not None:
                    #if len(line_pick[1]) != 2:
                        #print('wrong length for line ')
                        #print(line_pick[1])
 
        else:
            line_pick = min(line_set2, key = lambda x: line_len(x[0]['line']))
            affected.append((x,line_pick))
       
        #affected.append((x,line_set))

    print('total_lines',total_lines)
    return affected

def draw_trimmed_triangles(im,affected):
    """ supporting output for drawing triangle and its intersecting lines """
    for i, a in enumerate(affected):
        #if i not in range(5,15): continue
        #print('%dth triangle' % i)
        tri = a[0]
        lines = a[1]
        tri_c = tri['triangle']
        put_thing(im, tri_c, [0,0,255], tri['offset'])

        #for x in lines:
        l = lines[0]['line']
        #octr = lines[0]['ocontour']
        put_thing(im, l, [0,128,0], lines[0]['offset'], 2)
        l2 = lines[1]
        if l2 is not None:
            #print('intersects at %.2f degrs' % x[2])
            put_thing(im, l2, [128,12,0], (0,0), 1)
        else:
            put_thing(im, l, [128,128,0], lines[0]['offset'], 2)

def replace_trimmed_triangles(arr,triangles,affected, tri_tree, line_tree):
    """ supporting routine for triangle-line merging """
    for x in affected:
        combined = combine_features(arr,x[0],x[1][0])
        analyze_rectangles((combined,))
        analyze_triangles((combined,), arr)

        if triangle_passes(combined):
            triangles.append(combined)
            tri_tree.add_obj(combined)

            tri_tree.remove(x[0]['id'])
            if line_tree.has(x[1][0]['id']): line_tree.remove(x[1][0]['id'])


def update_list_against_tree(li, tree):
    """ Force coherency between list and rtree """
    new_list = []
    for x in li:
        if tree.has(x['id']): new_list.append(x)
    return new_list

def line_slope(line):
    """ return slope of line.  Infinity is clamped to a larger number. Large slopes are ceil'd to same large number."""
    dy = line[1][1] - line[0][1]
    dx = line[1][0] - line[0][0]
    if dx <.1 and dx>-.1:
        return MAX_SLOPE
    m = dy/dx
    if abs(m) > 50:
        return MAX_SLOPE
    return m

def slope_within(slop1,slop2,dv):
    return (slop1 < (slop2+dv)) and (slop1 > (slop2-dv))

def clamp_slopes(lines):
    """ Clamp lines below a threshold to be horizontal or vertical """
    for x in lines:
        s = x.slope
        l = x.abs_line
        if abs(s) < CLAMP_SLOPE_LOWER:
            x.slope = 0
            ydim = int(round((l[0][1] + l[1][1])/2))
            l[0][1] = ydim
            l[1][1] = ydim
            if l[0][1] != l[1][1]:
                print('horzontal line: ', l)

        if abs(s) > CLAMP_SLOPE_UPPER:
            x.slope = MAX_SLOPE
            xdim = int(round((l[0][0] + l[1][0])/2))
            l[0][0] = xdim
            l[1][0] = xdim

            if l[0][0] != l[1][0]:
                print('vertical line: ', l)



def coalesce_lines(arr,lines, tree):
    """ Combine lines that should be one line.  Organize lines into colinear groups and return. """
    def coalesce_add(item, group, masterlist):
        num_coalesced = 0
        for l in group:
            if l['coalesced']:
                num_coalesced += 1

        if num_coalesced == 0:
            for l in group:
                l['coalesced'] = True
            masterlist.append(group)
        else:
            if not item['coalesced']:
                item['coalesced'] = True
                masterlist.append([item])

    padding = 15
    ypara_groups = []
    xpara_groups = []
    diag_lines = []
    for x in lines:
        x['coalesced'] = False

    for x in lines:
        isvert = False
        slop = x.slope
        if slope_within(abs(slop),MAX_SLOPE,1):
            slop = MAX_SLOPE
            isvert = True
            center = x.abs_line[0][0]
            left = center - padding
            right= center + padding
            bottom = 0
            top = arr.shape[0]
        elif slope_within(slop,0,.20):
            slop = 0
            center = x.abs_line[0][1]
            bottom = center - padding
            top = center + padding
            left = 0
            right = arr.shape[1]
        else:
            diag_lines.append((x,slop))
            continue

        # get lines that cross infinite line
        para_lines = tree.intersectBox((left, bottom, right, top))

        # filter out perpendicular lines
        para_lines2 = [l for l in para_lines if slope_within(l['slope'], slop, .2)]

        coalesce_add(x,para_lines2,ypara_groups if isvert else xpara_groups)

    diag_groups = []

    # group the diagnol lines
    diag_lines = sorted(diag_lines, key = lambda x : line_len(x[0]['line']), reverse=True)
    for i,(x,slop1) in enumerate(diag_lines):
        margin = .2
        p0,p1 = x['abs-line'][0], x['abs-line'][1]

        # lines of same slope
        same_slops = [y for y,slop2 in diag_lines if  ((slop1+margin) > slop2 and (slop1-margin)<slop2)]
        # lines of close colinear-ness
        same_slops = [y for y in same_slops if collinear(p0,p1,y['abs-line'][0]) < 40]

        coalesce_add(x, same_slops, diag_groups)

    groups2 = []
    groups3 = []
    # sort lines
    for group in xpara_groups:
        groups2.append(sorted(group, key = lambda x : x['abs-line'][0][0]))
    for group in ypara_groups:
        groups2.append(sorted(group, key = lambda x : x['abs-line'][0][1]))
    for group in diag_groups:
        groups2.append(sorted(group, key = lambda x : line_len(((0,0),x['abs-line'][0]))))

    # walk groups of lines and find lines that should be merged
    for group in groups2:
        #if len(group)
        newgroup = []
        lastline = group[0]
        end = group[0]['abs-line'][1]
        for k,l in enumerate(group[1:]):

            end = lastline['abs-line'][1]
            if endpoints_connect(arr, end, l['abs-line'][0]):
                ext_line = (tuple(lastline['abs-line'][0]), tuple(l['abs-line'][1]))
                tree.remove(l['id'])
                tree.remove(lastline['id'])
                lastline = combine_features(arr, lastline, l, ext_line)

                analyze_rectangles((lastline,))

                inherit_from_line(lastline,l)
                lastline['line'] = np.array(ext_line)
                add_abs_line_detail((lastline,))

                tree.add_obj(lastline)
            else:
                newgroup.append(lastline)
                lastline = l

        newgroup.append(lastline)

        groups3.append(newgroup)

    # assign reference to colinear group
    for group in groups3:
        for l in group:
            l['colinear-group'] = group

    return groups3

def draw_para_lines(im, para_groups):
    """ output groups of lines to im image.  doesn't write to disk. """
    for i,para_lines in enumerate(para_groups):
        #if i in np.array([14]) :
        #if i in np.array([13]) or 1:

        #if i in range(13,15):
        #if i in np.array([10]):
            color = (randint(0,255),randint(0,255), randint(0,255))
            #if i in range(45,50):
                #assert(len(para_lines)>1)
            for x in para_lines:
                put_thing(im, x['line'], color, x['offset'], 2)

def bresenham_line(pt, m, b, sign):
    derr = abs(m)
    ysign = (1 if m >= 0 else -1)*sign
    error = 0.0
    x,y = pt
    while True:
        # plot(x,y)
        error += derr
        while error >= 0.5:
            y += ysign
            error -= 1.0
            yield (x,y)

        x += sign
        yield (x,y)


def extend_point(arr,pt, gen, lim, skips = 0, log = None):
    count = 0
    while True:
        pt[:] = next(gen)

        # break on limit
        if (pt[1] >= lim[0] or pt[0] >= lim[1]):
            if log is not None:
                print('break on LIM')
            break

        # break on N consecutive white pixels
        if arr[pt[1],pt[0]] != 0:
            if count >= skips:
                break
            count += 1
        else:
            count = 0

        if log is not None:
            log.append((np.copy(pt),arr[pt[1],pt[0]]))


    return count



def grow_lines(arr, lines):
    """ Grow lines along their slope if there are black pixels there """
    skip_count = 2
    arrT = np.transpose(arr)

    for i,x in enumerate(lines):
        
        m = x['slope']
        l = x['abs-line']
        b = l[0][1] - m*l[0][0]
        invert = False

        # invert vertical lines so rest of code can be used
        if m >= MAX_SLOPE:

            left = min((l[0],l[1]), key = lambda x : x[1])
            right = max((l[0],l[1]), key = lambda x : x[1])
            invert = True
            b = -b/m
            m = 0
            left[0],left[1] = left[1],left[0]
            right[0],right[1] = right[1],right[0]
            bim = arrT
        # no inversion
        else:
            bim = arr
            left = min((l[0],l[1]), key = lambda x : x[0])
            right = max((l[0],l[1]), key = lambda x : x[0])


        # use bresenham alg to get next pixel on line one by one
        leftdir = bresenham_line(left,m,b,-1)
        rightdir = bresenham_line(right,m,b,1)



        # extend left and right points
        lim = bim.shape
        lskips = extend_point(bim,left, leftdir, lim, skip_count)
        rskips = extend_point(bim,right, rightdir, lim, skip_count)


        # rewind white pixel steps
        leftdir = bresenham_line(left,m,b,1)
        rightdir = bresenham_line(right,m,b,-1)
        for j in range(0,lskips):
            left[:] = next(leftdir)
        for j in range(0,rskips):
            right[:] = next(rightdir)

        # invert line again to normal
        if invert:
            left[0],left[1] = left[1],left[0]
            right[0],right[1] = right[1],right[0]


def generate_line_girths(arr, lines):
    def perp_slope(m):
        if m == 0:
            return MAX_SLOPE
        if m >= MAX_SLOPE:
            return 0
        return -1.0/m

    arrT = np.transpose(arr)

    allpts = []
    for i,x in enumerate(lines):
        l = x['abs-line']
        m = x['slope']

        b = l[0][1] - m*l[0][0]
        invert = False

        left,right = l
        lim = arr.shape
        srcImg = arr

        if abs(m) >= 25:
            invert = True
            b = -b/m
            m = 0
            left[0],left[1] = left[1],left[0]
            right[0],right[1] = right[1],right[0]
            #print('  TRANPOSE')
            lim = arrT.shape
            srcImg = arrT

        
        #sign = 1 if m >= 0 else -1

        forwards = True

        if left[0] > right[0]:
            forwards = False
            sign = -1
        else:
            sign = 1

        colin = bresenham_line(left, m, b, sign)
        pts = []
        side1 = []
        side2 = []
        locs = []
        pm = perp_slope(m)
        if invert:
            x['pslope'] = 0
        else:
            x['pslope'] = pm

        count = 0
        while True:
            count += 1
            pt = next(colin)
            if pt[1] >= lim[0] or pt[0] >= lim[1]:
                break
            if pt[1] < 0 or pt[0] <0:
                break

            if forwards:
                if abs(pt[0]) > abs(right[0]):
                    break
            else:
                if abs(pt[0]) < abs(right[0]):
                    break

            perlin1 = bresenham_line(pt, pm, b, 1)
            perlin2 = bresenham_line(pt, pm, b, -1)

            #new_pts = []
            #if count % 9 == 0:
            l1 = len(pts)
            extend_point(srcImg,[],perlin2,lim,0,pts)
            l2 = len(pts)
            extend_point(srcImg,[],perlin1,lim,0,pts)
            l3 = len(pts)
            s1 = l2 - l1
            s2 = l3 - l2
            side1.append(s1)
            side2.append(s2)
            if invert:
                locs.append((pt[1],pt[0]))
            else:
                locs.append(pt)

        # invert line again to normal
        if invert:
            left[0],left[1] = left[1],left[0]
            right[0],right[1] = right[1],right[0]

            pts = [[pt[1],pt[0]] for pt in pts]

        pts.insert(0,left)
        allpts.append(pts)
        x['side-traces'] = (np.array(side1),np.array(side2))
        x['side-locs'] = locs

    return allpts

def remove_duplicate_lines(lines, tree):
    dups = []
    dist = 15
    for x in lines:
        l = x.abs_line
        #if x.id in [468,167,485]:
            #print('line',x.id)
            #print(x['boundxy'],x.offset,x['width'],x.height, 'slope:',x.slope)
            #print(tree.get_bounds(x))
            #continue
        #print(l)
        neighbors = tree.intersectPoint(l[0], dist)
        matches = []
        for pmatch in neighbors:
            l2 = pmatch.abs_line
            if (line_len((l2[0], l[1])) < dist) or (line_len((l2[1], l[1])) < dist):
                len1 = line_len(l)
                len2 = line_len(l2)
                if len1 < (len2+dist) and len1 > (len2-dist):
                    matches.append(pmatch)
        matches = sorted(matches, key = lambda k : k.id)
        if len(matches) > 1:
            #print('%d dups:' % x.id, [i.id for i in matches])
            dups.append(('-'.join([str(i.id) for i in matches]),matches))

    trash = []
    dupmap = {}
    for i in dups:
        if i[0] in dupmap:
            dupmap[i[0]][0] += 1
        else:
            dupmap[i[0]] = [1, len(i[1]), i[1]]

    for i in dupmap:
        count,length,matches = dupmap[i]
        if count == length:
            #print(i,'\n  = duplicate lines')
            # remove all but first
            trash += matches[1:]
        elif count < length:
            #print(i,'\n  = trash lines')
            # remove all
            trash += matches
        else:
            raise RuntimeError('Somehow got more dups than matches')

    trash = sorted(trash, key = lambda k : k.id)
    lastid = -1
    for t in trash:
        if t.id != lastid:
            lines.remove(t)
            tree.remove(t.id)
            t['trash'] = True
            lastid = t.id


def draw_pts(orig,pts_groups):
    for pts in pts_groups:
        for pt in pts:
            orig[pt[1], pt[0]] = (255,0,0)

def context_aware_correction(orig,ins):
    print('context_aware_correction')
    orig = np.copy(orig)

    arr = ins['arr']
    ocr = ins['ocr']
    circles = ins['circles']
    lines = ins['lines']
    add_abs_line_detail(lines)
    t1 = TIME()
    Updates.update(ins)
    tri_tree = RTree(arr.img.shape)
    line_tree = RTree(arr.img.shape, False)
    tri_tree.add_objs(ins['triangles'])
    line_tree.add_objs(ins['lines'])
    t2 = TIME()
    print('tree creation: %d ms' % (t2-t1))

    t1 = TIME()
    new_horz, new_verz = infer_ocr_groups(arr,ocr)
    t2 = TIME()
    print('ocr inferring time: %d ms' % (t2-t1))

    t1 = TIME()
    ins['circles'],ocr_rejects = untangle_circles(circles, new_horz + new_verz)

    new_horz = remove_ocr_groups(new_horz, ocr_rejects)
    new_verz = remove_ocr_groups(new_verz)
    t2 = TIME()
    print('circle untangle time: %d ms' % (t2-t1))


    t1 = TIME()
    triangles = ins['triangles']
    triangles,merges,merged = coalesce_triangles(arr['img'],triangles, tri_tree)
    ins['triangles'] = triangles
    t2 = TIME()
    print('triangle coalesce time: %d ms' % (t2-t1))

    #t1 = TIME()
    #affected = correct_trimmed_triangles(triangles, line_tree)
    #replace_trimmed_triangles(arr['img'],triangles,affected,tri_tree, line_tree)
    #ins['triangles'] = update_list_against_tree(triangles, tri_tree)
    #ins['lines'] = update_list_against_tree(lines, line_tree)
    #t2 = TIME()
    #print('triangle-line correction time: %d ms' % (t2-t1))

    #draw_trimmed_triangles(orig, affected)

    t1 = TIME()
    clamp_slopes(lines)
    merged = coalesce_lines(arr['img'],lines, line_tree)
    newlines = flatten(merged)
    print('COALESCE1: %d lines' % len(newlines))
    clamp_slopes(newlines)


    ins['lines'] = newlines
    t2 = TIME()
    #draw_para_lines(orig,merged)
    print('line coalesce time: %d ms' % (t2-t1))


    t1 = TIME()
    grow_lines(arr['img'],ins['lines'])
    Updates.lines(ins['lines'])
    line_tree = RTree(arr.img.shape, False)
    line_tree.add_objs(ins['lines'])
    t2 = TIME()
    print('line grow time: %d ms' % (t2-t1))

    t1 = TIME()
    remove_duplicate_lines(ins['lines'], line_tree)
    t2 = TIME()
    print('dup removal: %d ms' % (t2-t1))
    line_tree.test_coherency(ins['lines'])

    newmerged = []
    for group in merged:
        g = [l for l in group if not l['trash']]
        if len(g):
            newmerged.append(g)
    merged = newmerged


    t1 = TIME()
    allpts = generate_line_girths(arr['img'], ins['lines'])
    lines = ins['lines']
    colines = [Shape().init_from_line_group(g) for g in merged]
    #for x in lines:
    for x in colines:
        # TODO also look for --> <-- types via colinear groups
        syms = TriangleHumps.get_dimensions(x, im = orig)
    t2 = TIME()
    print('context-aware dimension detection: %d ms' % (t2-t1))

    #dump_plotly(ins['lines'], plotfuncs.side_traces)
    #dump_plotly(colines,plotfuncs.colinear_groups)
    #ins['lines'] = [l for l in ins['lines'] if l['id'] == 331]

    #draw_pts(orig,allpts)
    l = len(ins['lines'])

    #draw_ocr_group_rects(orig, new_horz, new_verz)
    save(orig,'output2.png')

    return ins


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: %s <x0> <y0> <x1> <y1>' % sys.argv[0])
        sys.exit(1)

    def bresenham_line(pt, m, b, sign):
        derr = abs(m)
        ysign = (1 if m >= 0 else -1)*sign
        error = 0.0
        x,y = pt
        while True:
            # plot(x,y)
            error += derr
            while error >= 0.5:
                y += ysign
                error -= 1.0
                yield (x,y)

            x += sign

    x0,y0,x1,y1 = [int(x) for x in sys.argv[1:]]
    
    line = ((x0,y0), (x1,y1))
    m = line_slope(line)
    b = line[0][1] - m*line[0][0]

    print(line[0],line[1])
    print('slope: %.2f' % m)

    g = (bresenham_line(line[0],m,b,-1))
    for i in range(0,int(m)+1):
        print(next(g))


