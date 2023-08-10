import math
import random

import numpy as np

import utils

#
# Functions
#
    
def generate(
        num_instances,
        num_targets = 4,
        seed = None, 
        window = None,
        polar = False,
        depot_pos = (0.5,0.5),
        C = 2.5,
    ):
    
    # Randomize
    if seed == None:
        seed = random.randint(0, 2**32)
    random.seed(seed)
    
    for index in range(num_instances):
        
        verts = []
        
        #
        # Depot
        #        
        
        depot = {'index': 0, 'x': depot_pos[0], 'y': depot_pos[1], 'w': 0}
        verts.append(depot)
      
        #
        # Targets
        #
        
        # Random positions
        
        for i in range(num_targets):
            target = {}
            target['index'] = i + 1
            
            # Coordinates
            
            if polar:
            
                # Regular polar
                theta = i / num_targets * 2 * math.pi
                target['x'] = math.cos(theta) / 2 + 0.5
                target['y'] = math.sin(theta) / 2 + 0.5
            
                # Random polar
                # theta = random.random() * 2 * math.pi
                # target['x'] = math.cos(theta) / 2 + 0.5
                # target['y'] = math.sin(theta) / 2 + 0.5
            
            else:    
            
                # Cartesian
                target['x'] = random.random()
                target['y'] = random.random()
                
            # Time Window
            
            # Default
            target['w'] = [6,3,2][i%3]
            
            # Uniform overide
            if window != None: target['w'] = window
                
            verts.append(target)
            
        #
        # Distances
        #
        
        dist = np.empty((num_targets+1, num_targets+1))
        for i,u in enumerate(verts):
            for j,v in enumerate(verts):
                dist[i,j] = math.dist((u['x'],u['y']), (v['x'],v['y']))
            
        #
        # Save
        #
        
        problem_data = verts, dist, C
        utils.save(problem_data, index)
        












