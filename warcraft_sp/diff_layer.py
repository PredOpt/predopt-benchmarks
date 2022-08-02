import numpy as np
import torch
from comb_modules.dijkstra import get_solver, shortest_pathsolution
from utils import maybe_parallelize

def  BlackboxDifflayer( lambda_val, neighbourhood_fn="8-grid"):
    solver = get_solver(neighbourhood_fn)
    class BlackboxDifflayer_cls(torch.autograd.Function):
        # def __init__(ctx, lambda_val, neighbourhood_fn="8-grid"):
        #     ctx.lambda_val = lambda_val
        #     ctx.neighbourhood_fn = neighbourhood_fn
        
        @staticmethod
        def forward(ctx, weights):
            ctx.weights = weights.detach().cpu().numpy()
            # ctx.suggested_tours = np.asarray (maybe_parallelize(solver, arg_list=list(ctx.weights)))
            # return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)
            ctx.suggested_tours = shortest_pathsolution(solver, weights)
            return ctx.suggested_tours
        @staticmethod
        def backward(ctx, grad_output):
            assert grad_output.shape == ctx.suggested_tours.shape
            grad_output_numpy = grad_output.detach().cpu().numpy()
            weights_prime = np.maximum(ctx.weights + lambda_val * grad_output_numpy, 0.0)
            better_paths = np.asarray(maybe_parallelize( solver, arg_list=list(weights_prime)))
            better_paths = torch.from_numpy(better_paths).float().to(grad_output.device)
            gradient = -(ctx.suggested_tours - better_paths) / lambda_val
            return   gradient #torch.from_numpy(gradient).to(grad_output.device)
    return BlackboxDifflayer_cls.apply


def  SPOlayer(  neighbourhood_fn="8-grid"):
    solver = get_solver(neighbourhood_fn)
    class SPOlayer_cls(torch.autograd.Function):
        # def __init__(ctx, lambda_val, neighbourhood_fn="8-grid"):
        #     ctx.lambda_val = lambda_val
        #     ctx.neighbourhood_fn = neighbourhood_fn
        
        @staticmethod
        def forward(ctx, weights, label, true_weights):
            ctx.save_for_backward(weights, label, true_weights)
            ctx.suggested_tours = shortest_pathsolution(solver, weights)
            return ctx.suggested_tours
        @staticmethod
        def backward(ctx, grad_output):
            weights, label, true_weights = ctx.saved_tensors
            spo_tour = shortest_pathsolution(solver, 2*weights - true_weights)
            
            gradient = (label - spo_tour)
            # assert grad_output.shape == ctx.suggested_tours.shape
            # grad_output_numpy = grad_output.detach().cpu().numpy()
            # weights_prime = np.maximum(ctx.weights + lambda_val * grad_output_numpy, 0.0)
            # better_paths = np.asarray(maybe_parallelize( solver, arg_list=list(weights_prime)))
            # better_paths = torch.from_numpy(better_paths).float().to(grad_output.device)
            # gradient = -(ctx.suggested_tours - better_paths) / lambda_val
            return   gradient, None, None #torch.from_numpy(gradient).to(grad_output.device)
    return SPOlayer_cls.apply
