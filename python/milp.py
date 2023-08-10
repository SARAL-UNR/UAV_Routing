import math

import numpy as np
import cvxpy as cp

#
# Functions
#

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
    
    #
    # MILP Decision Variables
    #
    
    # Arrival time
    ta = cp.Variable(len(dummy_verts), nonneg=True)
    
    # Remaining fuel on arrival
    y = cp.Variable(len(dummy_verts), nonneg=True)
    
    # Adjacency matrix
    x = cp.Variable(f.shape, boolean=True)
    
    #
    # MILP Objective
    #
    
    objective = cp.Minimize(cp.sum(cp.multiply(f, x)))
    
    # 
    # MILP Constraints
    # 
    
    constraints  = []
    
    # (2)
    # Start depot and targets have an outbound dergee of one
    for i in Sp+[0]:
        summation = []
        for j in Sp+F+[N+1]:
            if i != j:
                summation.append(x[i,j])
        constraints.append( 
            cp.sum(summation) == 1
        )
            
    # (3)
    # Fuel sites have an outbound dergee of zero or one
    for i in F:
        summation = []
        for j in Sp+F+[N+1]:
            if i != j:
                summation.append(x[i,j])
        constraints.append( 
            cp.sum(summation) <= 1
        )
         
    # (4)
    # Inbound degree equals oubound degree for targets and fuel sites
    for j in Sp+F:
        summation0 = []
        for i in Sp+F+[N+1]:
            if i != j:
                summation0.append(x[j,i])
        summation1 = []
        for i in Sp+F+[0]:
            if i != j:
                summation1.append(x[i,j])
        constraints.append( 
            cp.sum(summation0) - cp.sum(summation1) == 0
        )      
        
    # (5)
    # Travel time from vertex to vertex
    for i in Sp+F+[0]:
        for j in Sp+F+[N+1]:
            if i != j:
                constraints.append( 
                    ta[i] + f[i,j]*x[i,j] - te[0]*(1-x[i,j]) <= ta[j]
                )
        
    # (7)
    # Arrival time bounds
    for j in Vp:
        constraints += (
            ts[j] <= ta[j], 
            ta[j] <= te[j] 
        )
               
    constraints.append(y <= C)
        
    # (10)
    # Remaining fuel when traveling from a target to a vertex
    for i in Sp:
        for j in Sp+F+[N+1]:
            if i != j:
                constraints.append(
                    y[j] <= y[i] - (ta[j] - ta[i]) + C*(1-x[i,j])
                )
                
    # (11)
    # Remaining fuel when traveling from a depot/fuel to a target
    for i in [0]+F:
        for j in Sp:
            if i != j:
                constraints += [    
                    y[j] <= C - (ta[j] - ta[i]) + C*(1-x[i,j])
                ]
    
    # Remove double vertex sub-tours (i -> j NAND j -> i)
    constraints.append( 
        x + x.T <= 1  
    )
       
    # Ban travel from depot/fuel to depot/fuel
    for i in F+[0]:
        for j in F+[N+1]:
            constraints.append( 
                x[i,j] == 0  
            )
    
    #%% Solve
    
    prob = cp.Problem(objective, constraints)
    minimum = prob.solve(verbose=True)
    print()
    
    #%% Display
    
    # Edges
    dummy_edges = np.argwhere(x.value==1)
    edges = [(dummy_verts[i]['index'],dummy_verts[j]['index']) for i,j in dummy_edges]
           
    tour = [0]
    while tour[-1] != N+1:
        tour.append(np.where(x.value[tour[-1]])[0][0])
    
    ta = ta.value
    y = y.value
    x = x.value
    
    return dummy_edges, dummy_verts, edges, fnum, lcm, minimum, N, ta, targets, tour, x, y
    
    if __name__ == '__main__': globals().update(locals())

#%%

# if __name__ == '__main__': func()








