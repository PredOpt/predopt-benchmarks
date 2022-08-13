import numpy as np
import heapq
import torch
from functools import partial
from comb_modules.utils import get_neighbourhood_func
from collections import namedtuple
# from utils import maybe_parallelize

DijkstraOutput = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])


def dijkstra(matrix, neighbourhood_fn="8-grid", request_transitions=False):

    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)


def get_solver(neighbourhood_fn):
    def solver(matrix):
        return dijkstra(matrix, neighbourhood_fn).shortest_path

    return solver

# def shortest_pathsolution(solver, weights):
#     '''
#     solver: dijkstra solver
#     weights: torch tensor matrix
#     '''
#     np_weights = weights.detach().cpu().numpy()
#     suggested_tours = np.asarray (maybe_parallelize(solver, arg_list=list(np_weights)))
#     return torch.from_numpy(suggested_tours).float().to(weights.device)



# def growcache(solver, cache, output):
#     '''
#     cache is torch array [currentpoolsize,48]
#     y_hat is  torch array [batch_size,48]
#     '''
#     weights = output.reshape(-1, output.shape[-1], output.shape[-1])
#     shortest_path =  shortest_pathsolution(solver, weights).numpy() 
#     cache_np = cache.detach().numpy()
#     cache_np = np.unique(np.append(cache_np,shortest_path,axis=0),axis=0)
#     # torch has no unique function, so we need to do this
#     return torch.from_numpy(cache_np).float()

