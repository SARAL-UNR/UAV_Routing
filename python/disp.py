import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import networkx as nx

import utils

#
# Constants
#

figsize = (4.5, 4.5)

dpi = 200

cmap = plt.cm.gist_rainbow
# cmap = plt.cm.hsv
# cmap = plt.cm.viridis

#
# Tabulate
#

def table(dummy_verts, ta, y):
    table = np.array([list(v.values()) for v in dummy_verts])
    table = np.column_stack((table, ta, y))
    headers = [
        'index',
        'x',
        'y',
        'window',
        'dummy index',
        'start time',
        'end time',
        'arrival time', 
        'remaining fuel'
    ]
    print(tabulate(table, headers)+'\n')
    

def matrix(x, dummy_verts):
    stack = list(range(len(dummy_verts)))
    table = x
    table = np.char.mod('%d', table)
    table[table=='0'] = '.' 
    table = np.column_stack((stack, table))
    table = np.row_stack(([None]+stack, table))
    table = tabulate(table, tablefmt='plain', stralign='right')
    print(table+'\n')
    
#
# Matplotlib
#
    
def fuel(alg, verts, dummy_verts, tour, ta, y, minimum, C, N, targets, lcm):

    time    = []
    fuel    = []
    markers = []
    colors  = []

    for i in tour:
        time.append(ta[i])
        fuel.append(y[i])
        
        # Add a max fuel point for each visted refuel depot
        if dummy_verts[i]['index'] == 0:
            time.append(ta[i])
            fuel.append(C)
            markers += ['s','s']
            colors += [0,0]
        else:
            markers.append('o')
            colors.append(dummy_verts[i]['index'])
            
    # Remove first and last points of time and fuel
    # The first point is the UAV with no fuel
    # The last point is the UAV refueling
    time = [0] + time[1:-1] + [lcm]
    fuel = [C] + fuel[1:-1] + [0]

    # Plot fuel over time
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)    
    ax.set_title(alg+' Fuel Levels')
    ax.set_ylabel("Remaining Fuel")
    ax.set_xlabel('Time')
    ax.plot(time, fuel, zorder=0)
    for xp,yp,marker,c in zip(time,fuel,markers,colors):
        ax.scatter(
            xp, yp, marker=marker, c=c, vmin=0, vmax=len(targets)+1, cmap=cmap
        )    
    plt.show()


def timeline(alg, verts, dist, lcm, minimum, dummy_verts, ta, y, targets):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.invert_yaxis()
    ax.set_xlim(-.2, lcm+.2)
    ax.set_title(alg+' Timeline')
    ax.set_ylabel("Vertex Index")
    ax.set_xlabel('Time')
    
    # Display bars
    for i,v in enumerate(verts):
        
        # Vertex is the depot
        if i == 0: 
            width  = lcm 
            left   = 0
            index  = '0'
            labels = ['Depot']    
            color  = [cmap(0)]
                
        # Vertex is a target
        else:    
            width  = [v['w']] * (lcm // v['w']) 
            left   = width[0] * np.arange(len(width))
            index  = str(v['index'])
            labels = None
            color  = [cmap((i)/(len(targets)+1))] * (lcm // v['w'])
    
        # Plot the time windows
        rects = ax.barh(
            index, 
            width = width, 
            left = left,
            height = 0.5,
            color = color,
            edgecolor = 'w',
            linewidth = 3,
        )
    
        # Plot the time window's labels
        ax.bar_label(rects, labels=labels, label_type='center')
    
    # Plot vertical arival markings
    for i,v in enumerate(dummy_verts):
        index = v['index']
        # if index == 0: continue
        t = ta[i]    
        ax.plot((t, t), (index-.3, index+.3), 'k')
    
    plt.show()
    
#
# NetworkX
#

def route(alg, verts, dist, edges, minimum):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)    
    
    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    
    # Vertex positions
    pos = [(v['x'],v['y']) for v in verts]
    
    # Draw depot
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = [0],
        node_shape = 's',
        node_color = 'indigo',
    )
    
    # Draw targets
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = range(1,len(verts)),
        node_color = 'tab:blue',
    )
    
    # Draw vertex labels
    nx.draw_networkx_labels(
        G,
        pos, 
        font_color = "whitesmoke"
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle = "->",
        arrowsize = 10,
        # edge_color = range(G.number_of_edges()),
        # edge_cmap = plt.cm.gist_rainbow,
        width = 2,
        alpha = 0.5,
    )
    
    ax.set_title(alg+' Route ({:.3f} units)'.format(minimum))
    ax.set_xlim([-.05,1.05])
    ax.set_ylim([-.05,1.05])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    # plt.tight_layout()
    plt.show()


def dummy(alg, verts, dist, dummy_verts, edges, dummy_edges, minimum, N, fnum, targets):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)    
    
    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(dummy_verts)))
    G.add_edges_from(dummy_edges)
    
    # Vertex positions
    pos = nx.circular_layout(G)
    
    # Draw start and end depots
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = [0] + [N+1],
        node_shape = 's',
        node_color = [0, 0],
        cmap = cmap,
    )
    
    # Draw fuel depots
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = range(N+2,N+2+fnum),
        node_shape = 'h',
        node_color = cmap(0),
    )
    
    # Draw dummy targets
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = range(1,N+1),
        node_color = [dummy_verts[i]['index'] for i in range(1,N+1)],
        vmin = 0,
        vmax = len(targets)+1,
        cmap = cmap,
    )
    
    # Draw vertex labels
    nx.draw_networkx_labels(
        G,
        pos, 
        font_color = "whitesmoke"
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle = "->",
        arrowsize = 10,
        # edge_color = range(G.number_of_edges()),
        # edge_cmap = plt.cm.gist_rainbow,
        width = 2,
        alpha = 0.5,
    )
    
    ax.set_title(alg+' Dummy Verticies')
    # plt.tight_layout()
    plt.show()
    
#
# All Of The Above
#

def everything(alg, problem_data, solution_data):
    
    # Unpack problem and solution
    verts, dist, C = problem_data
    dummy_edges, dummy_verts, edges, fnum, lcm, minimum, N, ta, targets, tour, x, y = solution_data
    
    # matrix(x, dummy_verts)
    table(dummy_verts, ta, y)
    timeline(alg, verts, dist, lcm, minimum, dummy_verts, ta, y, targets)
    fuel(alg, verts, dummy_verts, tour, ta, y, minimum, C, N, targets, lcm)
    dummy(alg, verts, dist, dummy_verts, edges, dummy_edges, minimum, N, fnum, targets)
    route(alg, verts, dist, edges, minimum)
    
    
def saved(index, alg):
    problem_data = utils.load(index)
    solution_data = utils.load(index, alg)
    everything(alg, problem_data, solution_data)
    
    
    
