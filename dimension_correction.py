import math

import numpy as np
from scipy import stats

from analyzers import PARAMS
from cli import put_thing,line_len

class Dimension:
    def __init__(self,tri1,tri2,line):
        self.tri1 = tri1
        self.tri2 = tri2
        self.line = line

class TriangleHumps:
    """
        set of functions for detecting potential triangles along a line
    """
    IN = 0
    OUT= 1
    def get_humps(y, trigger):
        """
        detect "humps" in 1D array and return the ranges at which they occur in y.
        input: 1D array (typically the thickness of line at each pt), 
            trigger/threshold for detecting "humps"
        """
        humps = []
        lastmark = None
        markers = [None]*len(y)
        for i,val in enumerate(y):

            if lastmark is None: # look for rising edge
                if val > trigger:
                    if i != 0:
                        markers[i-1] = (i-1)
                        lastmark = i-1
                    else:
                        markers[i] = (i)
                        lastmark = i
            else:           # look for falling edge
                if val <= trigger:
                    markers[i] = (i)
                    hump = {'range': (lastmark,i),
                            'area': sum(y[lastmark:i]),
                            'length': i-lastmark}
                    humps.append(hump)
                    lastmark = None

        return humps,markers

    def group_humps(humps):
        """
            group together humps that are similar in area and length.
            Only returns groups made up of pairs.
        """
        groups = []
        added = False
        margin = 4
        for h1 in humps:
            a1 = h1['area']
            l1 = h1['length']
            for gru in groups:
                a2 = gru[0]['area']
                l2 = gru[0]['length']
                if a1 < (a2*1.15) and a1 > (a2*.85) or a1 < (a2+margin) and a1 > (a2-margin):
                    if l1 < (l2+margin) and l1 > (l2-margin):
                        gru.append(h1)
                        added = True
            if not added:
                groups.append([h1])
        groups = [g for g in groups if len(g) > 1 and ((len(g) & 1) == 0)]
        #groups = [g for g in groups if len(g) > 1 ]
        return groups

    def subtract_arrows(y,h1,h2):
        """
            returns difference of h1 - reverse(h2), accounts for length mismatch
        """
        p1,p2 = h1['range']
        y1 = y[p1:p2+1]
        p1,p2 = h2['range']
        y2 = y[p1:p2+1]
        y2 = y2[::-1]
        l = min(len(y1),len(y2))
        return y1[:l]-y2[:l]

    def get_hump_pairs(y,groups):
        """
            restructure data to be in pairs, add in diff_traces to store differnce for debugging.
        """
        pairs = []

        diff_traces = [[None]*len(y) for i in range(0,len(groups))]
        for i,gru in enumerate(groups):
            it = iter(gru)
            diff_trace = diff_traces[i]
            for hump1 in it:
                hump2 = next(it)
                pairs.append((hump1,hump2,diff_trace))
        return pairs,diff_traces


    def pass_symmetrical_pairs(y,pairs,):
        """
            decide if a pair is symmetrical enough.
            Return list of symmetrical pairs
        """
        # TODO filter out lines first.

        syms = []
        for hump1,hump2,diff_trace in pairs:
            diff = TriangleHumps.subtract_arrows(y,hump1,hump2)
            p1 = hump1['range'][0]
            area = hump1['area']
            diff_trace[p1:p1+len(diff)] = diff
            if sum(abs(diff))/len(diff) <= max(len(diff)/12,2.1) and area > 5:
                #print(sum(abs(diff))/len(diff),len(diff))
                # they should point out
                #if TriangleHumps.points(y,(hump1,hump2)) == TriangleHumps.OUT:
                #if 1:
                syms.append((hump1,hump2))
            else:
                #print(sum(abs(diff))/len(diff),len(diff))
                pass
        return syms

    def points(y,pair):
        """ indicate if a pair points <--out--> or -->in<--"""
        p1,p2 = pair[0]['range']
        arr1 = y[p1:p2+1]
        p1,p2 = pair[1]['range']
        arr2 = y[p1:p2+1]

        outside = sum(arr1[:len(arr1)//2]) + sum(arr2[len(arr2)//2:])
        inside = sum(arr1[len(arr1)//2:]) + sum(arr2[:len(arr2)//2])

        if inside < outside:
            return TriangleHumps.IN
        else:
            return TriangleHumps.OUT

    def get_dimensions(x, **kwargs):
        im = kwargs.get('im',None)
        s0 = np.array(x['side-traces'][0])
        s1 = np.array(x['side-traces'][1])
        dimensions = []

        y3 = s0 * s1

        mode = stats.mode(y3)[0][0]
        trigger = mode*2+1

        humps,markers = TriangleHumps.get_humps(y3,trigger)
        groups = TriangleHumps.group_humps(humps)
        pairs,diff_traces = TriangleHumps.get_hump_pairs(y3,groups)

        if 'debug_diffs' in kwargs:
            kwargs['debug_diffs'] += diff_traces
        if 'debug_groups' in kwargs:
            kwargs['debug_groups'] += groups
        if 'debug_markers' in kwargs:
            kwargs['debug_markers'] += markers

        syms = TriangleHumps.pass_symmetrical_pairs(y3,pairs)

        #if x.id in (50,):
            #print(x.id,'has %d humps,%d groups' % (len(humps), len(groups)))
            #print('  %d pairs, %d diff traces' % (len(pairs), len(diff_traces)))
            #print('  %d syms' % (len(syms)))
            #print('  ', pairs[0][:2])


        if len(syms):
            #print('line %d has %d pairs of potential triangles' %(x.id, len(syms)))
            #for p in syms:
                #dire = TriangleHumps.points(y3,p)
                #if dire == TriangleHumps.OUT:
                    #print('points out')
                #else:
                    #print('points in')
            locs = x['side-locs']
            pm = x['pslope']
            ss = s0+s1
            for a1,a2 in syms:
                ddir = TriangleHumps.points(y3,(a1,a2))
                p1,p2 = a1['range']
                p3,p4 = a2['range']

                if ddir == TriangleHumps.OUT:
                    base1 = locs[p2 + 0]
                    base2 = locs[p3 - 0]
                    tip1 = locs[max(p1 - 2,0)]
                    tip2 = locs[min(p4 + 2,len(locs)-1)]
                else:
                    base1 = locs[p1 + 0]
                    base2 = locs[p4 - 0]
                    tip1 = locs[max(p2 - 2,0)]
                    tip2 = locs[min(p3 + 2,len(locs)-1)]

                ang = math.atan(pm)
                dist = max(ss[p1:p2+1])/2.+2

                dx = int(dist * math.cos(ang))
                dy = int(dist * math.sin(ang))
                pm1 = (base1[0] - dx, base1[1] - dy)
                pp1 = (base1[0] + dx, base1[1] + dy)

                pm2 = (base2[0] - dx, base2[1] - dy)
                pp2 = (base2[0] + dx, base2[1] + dy)


                length = line_len((base1,tip1))

                # skip stubby triangles
                if (length/dist) < 0.5:
                    continue
                
                # skip infeasibly short dimensions
                if line_len((base1,base2)) < 5:
                    continue

                tri1 = (pm1,pp1,tip1)
                tri2 = (pm2,pp2,tip2)

                dim = Dimension(tri1,tri2,(tip1,tip2))
                dimensions.append(dim)

                #print(base_pt)
                if im is not None:
                    put_thing(im,tri1,(0,0,255),(0,0),1)
                    put_thing(im,tri2,(0,0,255),(0,0),1)
                    #put_thing(im,[base2],(0,0,255),(0,0),3)
        return dimensions

