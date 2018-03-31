import base64
from io import BytesIO

import numpy as np
from scipy import stats
from PIL import Image
import plotly
from plotly import tools
from plotly.graph_objs import Scatter,Layout

from analyzers import PARAMS
from cli import put_thing
from dimension_correction import TriangleHumps
from structures import Shape

class plotfuncs:

    def side_traces(x,im):
        """
            Show the girth of black pixels along a line.
            Highlight said line in orig image
        """
        s0 = x['side-traces'][0]
        s1 = x['side-traces'][1]
        t1 = Scatter(y=s0)
        t2 = Scatter(y=s1)
        y3 = s0 * s1
        mode = stats.mode(y3)[0][0]
        trigger = mode*2+1
        t3 = Scatter(y=y3)

        #put_thing(im,x['abs-line'],(255,0,0),(0,0),3)

        groups = []
        diff_traces = []
        markers = []
        TriangleHumps.get_dimensions(x,debug_groups=groups,debug_diffs=diff_traces,debug_markers = markers, im = im)

        annotations = []
        diff_traces = [Scatter(y=v) for v in diff_traces]
        t4 = Scatter(x=markers,y=[10]*len(markers),mode = 'markers+text')
        for gru in groups:
            for hump in gru:
                annotations.append({
                    'x':hump['range'][0],
                    'y':trigger,
                    'text':'%d,%d'%(hump['area'],hump['length']),
                    })

        name = 'mode=%d,trigger=%d,groups=%d' % (mode,trigger,len(groups))
        
        #return (t1,t2,t3,)
        #print('markers %d:' % x['id'],markers,[trigger]*len(markers))
        return [t3,t4,] + diff_traces,annotations, name

    def colinear_groups(coline,im):

        #x = Shape()
        #x.init_from_line_group(group)
        x = coline
        s0,s1 = x.side_traces
        p0,p1 = x.abs_line
        y3 = s0*s1

        for l in x.group:
            put_thing(im,l['abs-line'],(255,0,0),(0,0),3)

        #s0 = np.array([])
        #s1 = np.array([])
        #for x in group:
            #put_thing(im,x['abs-line'],(255,0,0),(0,0),3)
            #s0 = np.concatenate((s0,x['side-traces'][0]))
            #s1 = np.concatenate((s1,x['side-traces'][1]))
        #p0 = group[0]['abs-line'][0]
        #p1 = group[-1]['abs-line'][1]
        put_thing(im,(p0+(3,-3),p0),(0,0,255),(0,0),3)
        put_thing(im,(p1+(3,-3),p1),(0,0,255),(0,0),3)
        #y3 = s0 * s1
        #fakeline = {}
        t3 = Scatter(y=y3)

        return plotfuncs.side_traces(x,im)

        return [t3], None, '%d lines (%d,%d)->(%d,%d)' % (len(group),p0[0],p0[1],p1[0],p1[1])


def getbase64(nparr,):
    """
        get base64 string repr of object or np image
    """
    if type(nparr) == type({}):
        nparr = nparr['img']
    im = Image.fromarray(nparr)
    buf = BytesIO()
    im.save(buf,format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('ascii')



def _dump_plotly(objs, images, func):
    """
        make a plot of each object and put image next to it.
        func defines the type of plot and anything that is done to each image,obj pair
    """
    l = len(objs)
    #print(l)
    titles = []
    for i,x in enumerate(objs):
        if 'id' in x:
            titles.append('shape id %d' % x.id)
        else:
            titles.append('item %d' % i)
    fig = tools.make_subplots(rows=l, cols=1, subplot_titles = titles,print_grid=False )
    #print('figure attmpt: ')
    #fig['layout']['xaxis1'].update(title='monkeybar')
    #for x in fig['layout']['xaxis1']:
        #print(x)
    fig.layout.showlegend = False
    for i,x in enumerate(objs):
        traces,annotations,title = func(x,images[i])
        im = {
            "source": 'data:image/png;base64, ' + getbase64(images[i]),
            "x": 1,
            "y": 1 - i/(l-.5),
            "sizex": .5,
            "sizey": .5,
        }
        fig.layout.images.append(im)
        for t in traces:
            fig.append_trace(t,i+1,1)
        if title is not None:
            fig.layout['xaxis%d' % (i+1)].update(title=title)
        if annotations is not None:
            for a in annotations:
                a['xref'] = 'x%d' % (i+1)
                a['yref'] = 'y%d' % (i+1)
            fig.layout.annotations += annotations

    fig['layout'].update(height=400*l, width=1100, margin={
        'l':80,
        'r':320,
        't':100,
        'b':80,
        'pad':0,
        'autoexpand':True,
        },title='plots')

    return fig



def dump_plotly(objs, func = lambda x,y : x):
    # plotly can only handle 50 per html page for some reason
    images = [np.copy(PARAMS['orig']) for i in objs]
    for i in range(0,len(objs),50):
        fig = _dump_plotly(objs[i:i+50], images[i:i+50], func)
        plotly.offline.plot(fig, auto_open=True, filename='temp%d.html' % (i/50))


