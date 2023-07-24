import sys
import torch
import numpy as np
try:
    import ray
except ImportError as e:
    print(e)

def maybe_parallelize(function, arg_list):
    """
    Parallelizes execution is ray is enabled
    :param function: callable
    :param arg_list: list of function arguments (one for each execution)
    :return:
    """
    # Passive ray module check
    if 'ray' in sys.modules and ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]
def shortest_pathsolution(solver, weights):
    '''
    solver: dijkstra solver
    weights: torch tensor matrix
    '''
    np_weights = weights.detach().cpu().numpy()
    suggested_tours = np.asarray (maybe_parallelize(solver, arg_list=list(np_weights)))
    return torch.from_numpy(suggested_tours).float().to(weights.device)



def growcache(solver, cache, output):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    weights = output.reshape(-1, output.shape[-1], output.shape[-1])
    shortest_path =  shortest_pathsolution(solver, weights).numpy() 
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,shortest_path,axis=0),axis=0)
    # torch has no unique function, so we need to do this
    return torch.from_numpy(cache_np).float()