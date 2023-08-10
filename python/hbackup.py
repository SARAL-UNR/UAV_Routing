import math
from itertools import pairwise

import numpy as np

import graph

MAXINT = 2**31-1

def func(verts, dist, C):

#%% Setup Problem
    
    depot   = verts[0]
    targets = verts[1:]
    
    #
    # Transform Into Dummy Graph
    #
    
    dummy_verts = []
    
    # LCM of all time windows
    lcm = math.lcm(*(t['w'] for t in targets))
    
    # Starting depot
    dummy_depot = depot.copy()
    dummy_depot['dummy index'] = len(dummy_verts)
    dummy_depot['s'] = 0
    dummy_depot['e'] = lcm
    dummy_verts.append(dummy_depot)
    
    # Dummy targets
    for target in targets:
        for i in range(lcm // target['w']):
            dummy_target = target.copy()
            dummy_target['dummy index'] = len(dummy_verts)
            dummy_target['s'] = target['w'] *  i
            dummy_target['e'] = target['w'] * (i + 1)
            dummy_verts.append(dummy_target)
            
    # Number of targets
    N = len(dummy_verts) - 1
            
    # Number of fuel depots
    fnum = N - 1
    
    # End depot
    dummy_depot = depot.copy()
    dummy_depot['dummy index'] = len(dummy_verts)
    dummy_depot['s'] = 0
    dummy_depot['e'] = lcm
    dummy_verts.append(dummy_depot)
    
    # Refuel sites
    for i in range(fnum): 
        dummy_depot = depot.copy()
        dummy_depot['dummy index'] = len(dummy_verts)
        dummy_depot['s'] = 0
        dummy_depot['e'] = lcm
        dummy_verts.append(dummy_depot)
    
    #
    # Constants
    #
    
    # Target indicies
    Sp = list(range(1, N+1))
    
    # Fuel site indicies
    F = list(range(N+2, N+2 + fnum))
    
    # Vertex indicies
    Vp = list(range(len(dummy_verts)))
    
    # Distance matrix
    f = np.zeros((len(dummy_verts),len(dummy_verts)))
    for i,u in enumerate(dummy_verts):
        for j,v in enumerate(dummy_verts):
            if i != j:
                f[i,j] = dist[u['index'],v['index']]
    
    # Start times
    ts = [v['s'] for v in dummy_verts]
    
    # End times
    te = [v['e'] for v in dummy_verts]
    
    globals().update(locals())
    
    #%% Solve
    
def solve():
    
    global dummy_verts, ta, ts, te, f, N, Sp, F, Vp, lcm, targets, fnum
    
    starts = [dummy_verts[i]['s'] for i in Sp]
    ends   = [dummy_verts[i]['e'] for i in Sp]
    
    # Verticies ordered by end and start times
    order = sorted(zip(ends, starts, Sp))
    order = [i[2] for i in order]
    
    # Arrival times and remaining fuel for each vertex
    ta = [0] * len(dummy_verts)
    y = [0] * len(dummy_verts)
    
    # Initial depot
    i = 0
    
    # tour starts with initial depot
    tour = [i]
    
    # Stack for each vertex used for back tracking
    stack = {i:order.copy()[::-1] for i in order+[i]}
    
    # The next fuel depot to be visited
    cfuel = N+2
    
    j = None
    
    # Loop until all targets are part of the tour
    while len(set(Sp) - set(tour)) != 0:
        
        #
        # Find a valid j or backtrack i
        #
    
        if len(stack[i]) != 0:
            
            j = stack[i].pop()
            
            # j is not already in the tour
            if j in tour: 
                # count += 1
                # text  = f'{count:3d}: '
                # text += ' -> '.join(tour)
                # text += f' >< {j:3s} because it is in the tour'
                # print(text)
                continue
            
            # Path is not traversable
            if ta[i] + f[i,j] >= te[j]:
                # count += 1
                # text  = f'{count:3d}: '
                # text += ' -> '.join(tour)
                # text += f' >< {j:3s}'
                # print(text)
                continue
            
        # Backtrack i
        else:
            # count += 1
            # text  = f'{count:3d}: '
            # text += ' -> '.join(tour)
            # text += f' >< anything'
            # print(text)
            stack[i] = order.copy() # Reset the stack of i
            tour.pop()
            i = tour[-1]
            
            # Remove refueling
            if verts[i]['index'] == 0:
                cfuel -= 1
                tour.pop()
                i = tour[-1]
            
            continue
    
        # count += 1
        # text  = f'{count:3d}: '
        # text += ' -> '.join(tour)
        # text += f' -> {j:3s}'
        # print(text)
        
        # Time of arrival
        ta[j] = ta[i] + f[i,j]
        if ta[j] < ts[j]:
            ta[j] = ts[j]
        
        # Remaining fuel
        y[j] = y[i] - (ta[j] - ta[i])
        
        # Insufficient fuel
        if y[j] < f[j,0]:
            
            # Return to depot
            tour.append(cfuel)
            y[cfuel] = y[i] - f[i,cfuel] 
            ta[cfuel] = ta[i] + f[i,cfuel]
            
            # Travel to target
            y[j] = C - f[cfuel,j]
            ta[j] = ta[cfuel] + f[cfuel,j]
            if ta[j] < ts[j]:
                ta[j] = ts[j]
                
            cfuel += 1
            
        # Update vars
        tour.append(j)
        i = j
        
    # Return to the start of the tour
    j = N+1
    ta[j] = ta[i] + f[i,j]
    tour.append(j)    
    
    dummy_edges = list(pairwise(tour))
    edges = [(dummy_verts[i]['index'],dummy_verts[j]['index']) for i,j in dummy_edges]
    
    x = np.zeros((len(dummy_verts),len(dummy_verts)))
    
    # for dummy_edge in dummy_edges:
    minimum = x = None
    
    if __name__ == '__main__': globals().update(locals())
    
    return dummy_edges, dummy_verts, edges, fnum, lcm, minimum, N, ta, targets, tour, x, y
    

#%%    

if __name__ == '__main__': 
    verts, dist, C = graph.load(0)
    func(verts, dist, C)







