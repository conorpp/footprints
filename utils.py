import time,math,base64
from io import BytesIO
from scipy.signal import butter, lfilter, freqz
from numpy.linalg import norm
from PIL import Image, ImageDraw
import cv2
import numpy as np


# wraps np array image with metadata
counter = 0
def wrap_image(im,parent=None,offset=None):
    global counter
    specs = {
            'conf':0,           # % of pixels that are black under contour
            'area-ratio':0,     # a1/a2
            'a1':0,             # number of pixels in contour
            'a2':0,             # number of pixels
            'contour':[],        # inside rectangle growth
            'offset':[0,0],    # offset with respect to parent
            'img': im,          # image
            'height': 1,        # bounding rect height and width
            'width':1,
            'history':[],       # previous image
            'comment':'',
            'id': counter,

            'line-conf': 0,
            'aspect-ratio':0,
            'line-length':0,
            'length-area-ratio':0,
            'vertical':0,
            'sum':{'score':0.0, 'distinct':0, 'mode':[0,0], 'sum':[]},
            'rotated': False,

            'line-scan-attempt': 0,
            'line-estimates':[],
            'features':[],
            'traces':[],
            
            'merged':False,

            }

    counter = counter + 1
    if parent is not None:
        snapshot_img(specs,'parent',parent)

    if offset is not None:
        specs['offset'][0] += offset[0]
        specs['offset'][1] += offset[1]

    return specs

def count_black(x):
    nz = np.count_nonzero(x)
    return x.shape[0] * x.shape[1] - nz


def snapshot_imgs(imgs,comment,parent=None):
    for x in imgs:
        snapshot_img(x,comment,parent)

def snapshot_img(im,comment,parent=None):
    cop = {}
    if parent is None:
        for key, value in im.items():
            cop[key] = value
        im['offset'] = im['offset'][:]
        im['img'] = np.copy(im['img'])
    else:
        for key, value in parent.items():
            cop[key] = value
        im['offset'] = parent['offset'][:]
        im['history'] += parent['history']


    cop['comment'] = comment
    im['history'].append(cop)



def remove_alpha(im):
    if len(im.shape)>2 and im.shape[2] == 4:
        return np.delete(im, 3, 2)
    return im

def greyscale(im):
    if len(im.shape) > 2:
        if im.shape[2] == 4:
            im = np.delete(im, 3, 2)
        if im.shape[2] == 3:
            im = np.delete(im, 2, 2)
        if im.shape[2] == 2:
            im = np.delete(im, 1, 2)
        im = im.reshape(im.shape[:2])
    return im

def color(arr):
    if len(arr.shape) < 3:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
    if arr.shape[2] != 1: raise ValueError('needs to be single channel')
    arr = np.tile(arr, (1,1,3))
    return arr

def load_image(name):
    im = Image.open(name)
    return np.array(im,dtype=np.uint8)

#def preprocess(im):

# checks two points and 5 midpoints
def still_inside(c,p1,p2):
    p1 = (p1[0],p1[1])
    p2 = (p2[0],p2[1])

    p3 = ((p1[0]/2 + p2[0]/2),(p1[1]/2 + p2[1]/2))
    p4 = ((p1[0]*1/5 + p2[0]*4/5),(p1[1]*1/5 + p2[1]*4/5))
    p5 = ((p1[0]*2/5 + p2[0]*3/5),(p1[1]*2/5 + p2[1]*3/5))
    p6 = ((p1[0]*3/5 + p2[0]*2/5),(p1[1]*3/5 + p2[1]*2/5))
    p7 = ((p1[0]*4/5 + p2[0]*1/5),(p1[1]*4/5 + p2[1]*1/5))
    corners = (cv2.pointPolygonTest(c, p1,0) > 0 ) and (cv2.pointPolygonTest(c, p2,0) > 0 )
    with_mid = corners and (cv2.pointPolygonTest(c, p3,0) > 0 )
    with_mid = with_mid and (cv2.pointPolygonTest(c, p4,0) > 0 )
    with_mid = with_mid and (cv2.pointPolygonTest(c, p5,0) > 0 )
    with_mid = with_mid and (cv2.pointPolygonTest(c, p6,0) > 0 )
    with_mid = with_mid and (cv2.pointPolygonTest(c, p7,0) > 0 )
    return with_mid

def point_in_contour(c,p):
    return cv2.pointPolygonTest(c, p,0) > 0

def point_ineq_contour(c,p):
    return cv2.pointPolygonTest(c, p,0) >= 0


def centroid(c):
    moms = cv2.moments(c)
    x = int(moms['m10']/moms['m00'])
    y = int(moms['m01']/moms['m00'])
    return x,y

def show(im):
    tmp = Image.fromarray(im)
    tmp.show()

def getbase64(nparr,):
    if type(nparr) == type({}):
        nparr = nparr['img']
    im = Image.fromarray(nparr)
    buf = BytesIO()
    im.save(buf,format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('ascii')


def save(nparr,name):
    if type(nparr) == type({}):
        nparr = nparr['img']
    im = Image.fromarray(nparr)
    im.save(name)

def trace_sum(im,contour):

    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.drawContours(mask,[contour],0,255,1)
    pixelpoints = np.transpose(np.nonzero(mask))

    total = 0
    for i,j in pixelpoints:
        total += (im[i,j] == 0)

    return total, len(pixelpoints)

def line_conf(img,side):
    s,l = line_sum(img, side), line_len(side)
    if l == 0:
        return 0.0
    else:
        return (1.0 - float(s)/l)



def rect_confidence(im,con):
    lines = []

    for i in range(0,4):
        s,l = line_sum(im,con[0+i:2+i])/255, line_len(con[0+i:2+i])
        if l == 0:
            lines.append(0.0)
        else:
            lines.append(1.0 - float(s)/l)

    return np.array(lines)

def circle_confidence(im,con):
    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.circle(mask,con[0],con[1],255,1)
    pixelpoints = np.transpose(np.nonzero(mask))

    total = 0
    for i,j in pixelpoints:
        total += (im[i,j] == 0)

    return float(total)/len(pixelpoints)

def encircle(img,cnt,**kwargs):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    offset = kwargs.get('offset',[0,0])
    center = (int(x) + offset[0],int(y) + offset[1])
    radius = int(radius)
    cv2.circle(img,center,int(radius * 3),(0,255,0),2)

def print_img(x, itr=None, **kwargs):
    ty = kwargs.get('shape','line')
    if ty == 'rect':
        s = '%d: %.3f, a1: %d, a2: %d, area-ratio: %.4f, contour-area: %.4f wxh: %dx%d %s' % (
                x['id'], x['conf'], x['a1'], 
                x['a2'], x['area-ratio'], x['contour-area'],
                x['width'],x['height'],
                x['comment']
                )
 
    else:
        s = '%d: %.3f, ar: %.2f, vert: %d, score: %.2f, sum-len: %d, distinct: %d, mode: %d, pixels: %d, wxh: %dx%d %s' % (
                x['id'], x['line-conf'], x['aspect-ratio'], 
                x['vertical'], x['sum']['score'],
                len(x['sum']['sum']), x['sum']['distinct'],
                x['sum']['mode'][0], count_black(x['img']),
                x['width'],x['height'],
                x['comment']
                )

    if itr is not None:
        s = '('+str(itr)+') ' + s
    print(s)

def save_history(x):
    print('saving history for image %d' % x['id'])
    for i,y in enumerate(x['history']):
        print_img(y,i)
        name = 'img%d-%d.png' % (x['id'],i)
        save(y['img'],'hist/'+name)
    print_img(x,'current')
    name = 'img%d-%d.png' % (x['id'],i+1)
    save(x['img'],'hist/'+name)


def polarize(arr):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    (thresh, arr) = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return arr


def scan_dim(im,dim):
    return np.sum(im == 0,axis=dim)

def scan_trim(arr):
    return np.trim_zeros(arr)


def convert_rect_contour(c):
    # preproc: [tl,tr,br,bl,tl]
    # old:     [br,tr,tl,bl,br]
    return np.copy([c[2], c[1], c[0], c[3], c[2]])


def set_line(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    arr[y1:y2+1,x1:x2+1] = 0

def line_exists(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    s = np.sum(arr[y1:y2+1,x1:x2+1])

    return s == 0

def line_slice(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    return arr[y1:y2+1,x1:x2+1]


def line_sum(arr,line):
    x1 = min(line[0][0],line[1][0])
    x2 = max(line[0][0],line[1][0])
    y1 = min(line[0][1],line[1][1])
    y2 = max(line[0][1],line[1][1])

    s = np.sum(arr[y1:y2+1,x1:x2+1])

    return s

def line_vert(line):
    return 1 if line[0][0] == line[1][0] else 0

def line_horz(line):
    return 1 if line[0][1] == line[1][1] else 0




def line_len(line):
    p0 = line[0]
    p1 = line[1]
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def timestamp(): return int(round(time.time() * 1000))
def TIME(): return int(round(time.time() * 1000))


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def test_for_dups(pdups):
    """ Check for duplicates in a python list by eye """
    for l1 in pdups:
        count = 0
        for l2 in pdups:
            if l1 is l2:
                count += 1
        print('---count: %d---' % count)
def flatten(groups):
    """ Flatten a 2D python list """
    return [x for group in groups for x in group]



def endpoints_connect(arr,p1,p2):
    """  Determine if two points are connected by mostly black pixels in straight line  """
    if line_len((p1, p2)) < 3:
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

    #print(black_count/d)
    if black_count/d >= .8:
        return True
    return False

def collinear(p1, p2, p3):
    """ return perpendicular distance between p1,p2 line and p3 point """
    return norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)


