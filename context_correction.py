from scipy import stats
from utils import *
from filters import *
from analyzers import *
from processors import *
import preprocessing

def group_ocr(ocr, dim=0):
    # dim=0 means horizontal text, dim=1 means vertical text
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
            if len(group) > 1:
                ocr_groups.append(group)
            group = [x]
            starty = y
    if len(group) > 1:
        ocr_groups.append(group)

    print('ocr_groups:', len(ocr_groups))

    ocr_groups = [sorted(x, key = lambda i : i[indx][dim]) for x in ocr_groups]
    ocr_groups2 = []
    SEPARATING_DIST = 4 # TODO normalize

    # Segment into consecutive groups on other dimension
    for i,group in enumerate(ocr_groups):
        print('group',i)
        x1 = group[0]
        segment = [x1]

        for x2 in group[1:]:
            dx = (x2[indx][dim] - x1[indx][dim] - x1['width'])
            print('dx: %d' % dx)

            if abs(dx) > SEPARATING_DIST:
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


def context_aware_correction(orig,ins):
    print('context_aware_correction')
    orig = np.copy(orig)
    ocr = ins['ocr']

    #print('by width')
    #ocr = sorted(ocr, key = lambda x : x['width'])
    #for x in ocr:
        #print (x['width'], x['height'])

    # get starting x,y for bottom left coord, with added offset
    for x in ocr:
        x['boundxy2'] = (x['offset'][0] + x['boundxy'][0], x['offset'][1] + x['boundxy'][1] + x['height'])
        x['boundxy3'] = (x['offset'][0] + x['boundxy'][0] + x['width'], x['offset'][1] + x['boundxy'][1])

    #vert,horiz = if_rotated(ocr)
    horz = group_ocr(ocr,0)
    verz = group_ocr(ocr, 1)

    new_horz,new_verz = block_redundant_groups(horz,verz)

    ocr_groups = new_horz+new_verz
    print(len(ocr_groups),'groups')
    for group in ocr_groups:
        leftest = group[0]
        rightest = group[-1]

        pt1 = (leftest['boundxy2'][0], leftest['boundxy2'][1] - leftest['height'])
        pt2 = (rightest['boundxy2'][0] + rightest['width'], rightest['boundxy2'][1])

        cv2.rectangle(orig, pt1,pt2, [0,0,255])

    save(orig,'output2.png')

    return ins



