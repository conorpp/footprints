from scipy import stats
from utils import *
from filters import *
from analyzers import *
from processors import *
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
    # block characters that are largely different from the group
    #for group in ocr_groups:
        #if odim:
            #group = [x for x in group if (x['width'] > wmode*.2 and x['width'] < wmode*1.5)]
        #else:
            #group = [x for x in group if (x['height'] > hmode*.2 and x['height'] < hmode*1.5)]
        #if len(group):
            #ocr_groups2.append(group)

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

def context_aware_correction(orig,ins):
    print('context_aware_correction')
    orig = np.copy(orig)
    arr = ins['arr']
    ocr = ins['ocr']
    #feat_avg = avg_rect_area(ins['rectangles'], ins['irregs'])
    #ocr = [x for x in ocr if (x['width'] * x['height']) < feat_avg]
    #print('feature average area is ', feat_avg)


    #print('by width')
    #ocr = sorted(ocr, key = lambda x : x['width'])
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

                #not_rotated.append(x)

    #rotate_right(not_rotated)
    #analyze_ocr(not_rotated)
    #rotate_left(not_rotated)


    ocr_groups = new_horz+new_verz
    print(len(ocr_groups),'groups')
    for i,group in enumerate(new_horz):
        leftest = group[0]
        rightest = group[-1]

        pt1 = (leftest['boundxy2'][0], leftest['boundxy2'][1] - leftest['height'])
        pt2 = (rightest['boundxy2'][0] + rightest['width'], rightest['boundxy2'][1])

        cv2.rectangle(orig, pt1,pt2, [0,0,255])
        s = ''.join(x['symbol'] for x in group)
        print(s)
        #print(pt1, pt2)
        #if i == 9:
            #break
    for i,group in enumerate(new_verz):
        leftest = group[0]
        rightest = group[-1]

        pt1 = (leftest['boundxy2'][0], leftest['boundxy2'][1] - leftest['height'])
        pt2 = (rightest['boundxy2'][0] + rightest['width'], rightest['boundxy2'][1])

        cv2.rectangle(orig, pt1,pt2, [0,180,0])
        s = ''.join(x['symbol'] for x in group)[::-1]
        print(s)
        #if i == 4:
            #break



    save(orig,'output2.png')

    return ins



