import glob
import numpy as np
import pickle

#
# Constants
#

MAXINT = 2**31-1

problem_folder = '../saves/problems/'
solution_folder = '../saves/'

#
# Functions
#

def load(index:int, alg:str=None):
    """
    Returns a list of all variables for a specific problem or solution instance.

    Parameters
    ----------
    index : int
        The problem's index.
    alg : str, optional
        The solution's algorithm. Returns solution data if included

    """
    if alg == None:
        path = problem_folder + str(index) + '.pkl'
    else:
        path = solution_folder + alg.lower() + '/' + str(index) + '.pkl'
    with open(path, 'rb') as file:
        return pickle.load(file)
        
    
def save(data, index:int, alg:str=None):
    if alg == None:
        path = problem_folder + str(index) + '.pkl'
    else:
        path = solution_folder + alg.lower() + '/' + str(index) + '.pkl' 
    with open(path, 'wb') as file:
        return pickle.dump(data, file)
    
        
        
def num_saved():
    """
    Returns the number of saved problem instances.
    """
    return len(glob.glob(problem_folder+'*.pkl'))

