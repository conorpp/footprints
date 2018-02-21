import math
from random import randint


from scipy import stats
from utils import *
from filters import *
from analyzers import *
from processors import *
from cli import put_thing
from structures import RTree
import preprocessing

def group_ocr_lines(ocr,dim=0):
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

    print('ocr_groups:', len(ocr_groups))

    ocr_groups = [sorted(x, key = lambda i : i[indx][dim]) for x in ocr_groups]
    return ocr_groups

def wrap_image_ocr(image,parent, char, conf, tl, h, w):
    image = wrap_image(image,parent)
    image['symbol'] = char
    image['ocr-conf'] = conf
    image['boundxy'] = tl
    image['height'] = h
    image['width'] = w
    return image

def detect_punct(arr, x1, x2, dx, dim, ):

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

def if_rotated(inps):
    rot = []
    nope = []
    for x in inps:
        if x['rotated']:
            rot.append(x)
        else:
            nope.append(x)
    return rot, nope

def group_crosses(group1,group2):
    for x in group1:
        for y in group2:
            if x['id'] == y['id']:
                return True
    return False
def item_contained(x,group_set):
    for group2 in group_set:
        for y in group2:
            if x['id'] == y['id']:
                return True
    return False



def block_redundant_groups(horz, verz):
    new_horz = []
    new_verz = []
    
    for group in horz:
        if len(group)>1:
            new_horz.append(group)
 
    for group in verz:
        if len(group)>1:
            new_verz.append(group)
    print(len(verz+horz),'initial groups')

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

def avg_rect_area(rects, irregs):
    tot = 0.0
    count = 0
    for x in rects:
        tot += x['a1']
        count += 1
    for x in irregs:
        tot += x['a1']
        count += 1
    return tot/count

def infer_ocr_groups(arr, ocr):
    widths = []
    heights = []
    for x in ocr:
        widths.append(x['width'])
        heights.append(x['height'])

    wmode = stats.mode(widths)[0][0]
    hmode = stats.mode(heights)[0][0]
    mode = max(wmode,hmode,35)

    print(wmode)
    print(hmode)

    ocr = [x for x in ocr if x['width'] < (mode*2)]
    ocr = [x for x in ocr if x['height'] < (mode*2)]

    # get starting x,y for bottom left coord, with added offset
    for x in ocr:
        x['boundxy2'] = (x['offset'][0] + x['boundxy'][0], x['offset'][1] + x['boundxy'][1] + x['height'])
        x['boundxy3'] = (x['offset'][0] + x['boundxy'][0] + x['width'], x['offset'][1] + x['boundxy'][1])

    #vert,horiz = if_rotated(ocr)
    print('horz:')
    horz = group_ocr(arr,ocr, 0)
    print('verz:')
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

    newim = wrap_image(bg, t1)
    newim['offset'] = np.array((0,0))
    return newim



def coalesce_triangles(arr,triangles, tree):
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
        print(t1['id'])
        print(t2['id'])
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

def update_bounding_boxes(outs):
    for x in outs['circles']:
        center,r = x['circle']
        xmin = center[0]-r
        ymin = center[1]-r
        x['boundxy'] = (xmin,ymin)
        x['width'] = 2 * r
        x['height'] = 2 * r
        x['offset'] = np.array(x['offset'])
        #x['offset'] = np.array(x['offset'])
        #x['offset'] = np.reshape(np.array(x['offset']),(2))
    for x in outs['triangles']:
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
    for x in outs['lines']:
        xmin = min(x['line'][0][0], x['line'][1][0],)
        xmax = max(x['line'][0][0], x['line'][1][0],)
        ymin = min(x['line'][0][1], x['line'][1][1],)
        ymax = max(x['line'][0][1], x['line'][1][1],)
        if 'ocontour' not in x:
            analyze_rectangles((x,))
        x['boundxy'] = (xmin,ymin)
        x['width'] = xmax-xmin
        x['height'] = ymax-ymin
        x['offset'] = np.array(x['offset'])
        #x['offset'] = np.reshape(np.array(x['offset']),(2))
    for x in outs['rectangles']:
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

    #for i in ('rectangles', 'lines', 'circles', 'triangles','irregs','ocr'):
        #for x in outs[i]:
            #absoffset = x['']

def line_intersection(L1, L2):
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

# Return true if line segments AB and CD intersect
def does_intersect(l0,l1):
    A,B,C,D = l0[0], l0[1], l1[0], l1[1]
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def angle_between_lines(l0,l1):
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
    new_list = []
    for x in li:
        if tree.has(x['id']): new_list.append(x)
    return new_list

def line_slope(line):
    dy = line[1][1] - line[0][1]
    dx = line[1][0] - line[0][0]
    if dx <.1 and dx>-.1:
        return 1000
    m = dy/dx
    if abs(m) > 50:
        return 1000
    return m

def slope_within(slop1,slop2,dv):
    return (slop1 < (slop2+dv)) and (slop1 > (slop2-dv))

def coalesce_lines(arr,lines, tree):
    def endpoints_connect(arr,p1,p2):
        if line_len((l['abs-line'][0], end)) < 7:
            return True
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        d = int(max(dx,dy))
        xs = np.linspace(p1[0], p2[0],d)
        ys = np.linspace(p1[1], p2[1],d)
        black_count = 0
        for i in range(0,d):
            x = int(xs[i])
            y = int(ys[i])
            black_count += (arr[y,x] == 0)
        if black_count/line_len((p1,p2)) > .9:
            #print('---black pixels connect!---')
            return True


    padding = 5
    ypara_groups = []
    xpara_groups = []
    print('%d lines colaescing' % len(lines))
    for x in lines:
        x['coalesced'] = False
    for x in lines:
        isvert = False
        slop = line_slope(x['line'])
        if slope_within(abs(slop),1000,1):
            #print('vert line')
            isvert = True
            center = x['abs-line'][0][0]
            #center = x['line'][0][1]
            # left, bottom, right, top
            left = center - padding
            right= center + padding
            bottom = 0
            top = arr.shape[0]
        elif slope_within(slop,0,.15):
            #print('horz line')
            center = x['abs-line'][0][1]
            #center = x['line'][0][0]
            # left, bottom, right, top
            bottom = center - padding
            top = center + padding
            left = 0
            right = arr.shape[1]
        else:
            print('ignoring non-horz non-vert line %.2f' % (slop))
            xpara_groups.append((x,))
            continue

        # get lines that cross infinite line
        para_lines = tree.intersectBox((left, bottom, right, top))

        # filter out perpendicular lines
        para_lines2 = [l for l in para_lines if slope_within(line_slope(l['line']), slop, .2)]
        para_lines2.append(x)
        
        num_coalesced = 0
        for l in para_lines2:
            if l['coalesced']:
                num_coalesced += 1
        if num_coalesced <= int(len(para_lines2)/2):
            for l in para_lines2:
                l['coalesced'] = True
            if isvert:
                ypara_groups.append(para_lines2)
            else:
                xpara_groups.append(para_lines2)

    groups2 = []
    # sort lines
    for group in xpara_groups:
        groups2.append(sorted(group, key = lambda x : x['abs-line'][0][0]))
    for group in ypara_groups:
        groups2.append(sorted(group, key = lambda x : x['abs-line'][0][1]))

    groups3 = []
    for group in groups2:
        #print('end',group[0])
        newgroup = []
        #end = max(group[0]['line'][1],group[0]['line'][0], key = lambda x : x[0])
        end = group[0]['abs-line'][1]
        lastline = group[0]
        for l in group[1:]:

            end = lastline['abs-line'][1]

            if endpoints_connect(arr, end, l['abs-line'][0]):
                lastline = combine_features(arr, lastline, l, (tuple(lastline['abs-line'][0]), tuple(l['abs-line'][1])))
                #lastline = l
                analyze_rectangles((lastline,))
                analyze_lines((lastline,))
                add_abs_line_detail((lastline,))
            else:
                newgroup.append(lastline)
                lastline = l

        newgroup.append(lastline)

        groups3.append(newgroup)


    print(len(groups3), 'groups3')
    return groups3

def draw_para_lines(im, para_groups):
    for i,para_lines in enumerate(para_groups):
        if i in np.array([0,1,2])+33 or 1:
            color = (randint(0,255),randint(0,255), randint(0,255))
            #if i in range(45,50):
                #assert(len(para_lines)>1)
            for x in para_lines:
                put_thing(im, x['line'], color, x['offset'], 2)




def context_aware_correction(orig,ins):
    print('context_aware_correction')
    orig = np.copy(orig)
    

    arr = ins['arr']
    ocr = ins['ocr']
    circles = ins['circles']
    lines = ins['lines']
    add_abs_line_detail(lines)
    t1 = TIME()
    update_bounding_boxes(ins)
    tri_tree = RTree(arr['img'])
    line_tree = RTree(arr['img'])
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
    merged = coalesce_lines(arr['img'],lines, line_tree)
    t2 = TIME()
    draw_para_lines(orig,merged)
    print('line coalesce time: %d ms' % (t2-t1))



    #for x in triangles:
        #put_thing(orig, x['triangle'], [255,0,0], x['offset'])
    ##for x in merges:
        ##put_thing(orig, x[0]['triangle'], [255,0xc0,0], x[0]['offset'])
        ##put_thing(orig, x[1]['triangle'], [255,0xc0,0], x[1]['offset'])
    #for x in merged:
        #put_thing(orig, x['triangle'], [0,0,255], x['offset'])

    #draw_ocr_group_rects(orig, new_horz, new_verz)
    save(orig,'output2.png')

    return ins



