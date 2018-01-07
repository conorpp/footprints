
from PIL import Image, ImageDraw
import cv2
import numpy as np


# wraps np array image with metadata
counter = 0
def wrap_image(im,parent=None):
    global counter
    specs = {
            'conf':0,           # % of pixels that are black under contour
            'area-ratio':0,     # a1/a2
            'a1':0,             # number of pixels in contour
            'a2':0,             # number of pixels
            'contour':0,        # inside rectangle growth
            'offset':[0,0],     # offset with respect to parent
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
            }

    counter = counter + 1
    if parent is not None:
        snapshot_img(specs,'parent',parent)

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

def still_inside(c,p1,p2):
    p1 = (p1[0],p1[1])
    p2 = (p2[0],p2[1])
    return (cv2.pointPolygonTest(c, p1,0) > 0 ) and (cv2.pointPolygonTest(c, p2,0) > 0 )

def centroid(c):
    moms = cv2.moments(c)
    x = int(moms['m10']/moms['m00'])
    y = int(moms['m01']/moms['m00'])
    return x,y

def show(im):
    tmp = Image.fromarray(im)
    tmp.show()

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

def rect_confidence(im,con):
    s,t = trace_sum(im,con)
    return float(s)/t

def encircle(img,cnt,**kwargs):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    offset = kwargs.get('offset',[0,0])
    center = (int(x) + offset[0],int(y) + offset[1])
    radius = int(radius)
    cv2.circle(img,center,int(radius * 3),(0,255,0),2)

def print_img(x, itr=None):
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


def polarize(arr):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    (thresh, arr) = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return arr


