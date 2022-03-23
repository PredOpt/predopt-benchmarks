import numpy as np
import math
import pickle as pkl
import argparse

# Reported Experiments in Appendix D:
# n_samples = {100,1000}
# deg = {1,4,8,16}
# tau = {1,2}

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--p", type=int, default=5)
parser.add_argument("--tau", type=int, default=0.1)
parser.add_argument("--deg", type=int, default=1)
args = parser.parse_args()

n_samples = args.n_samples
n = args.n    # number of assets
p = args.p    # number of features
tau = args.tau  # noise level parameter
deg = args.deg  # degree parameter

# Bernoulli(0.5) matrix
# parameters of 'true' model
B = np.int_(np.random.rand(n,p) > 0.5)

# load factor matrix
# 50 and 4 are specified in the paper
L =   2*0.0025*tau*np.random.rand(n,4) - 0.0025*tau


pairs = []
for _ in range(0,n_samples):
    x = np.random.normal(0, 1, size=p)    # feature vector - standard normal
    r = (  (0.05/math.sqrt(p)) * np.matmul(B,x) + (0.1)**(1/deg)  )**(deg)
    f = np.random.normal(0, 1, size=4)
    eps = np.random.normal(0, 1, size=n)
    r = r + np.matmul(L,f) + 0.01*tau*eps
    c = -r
    pairs.append(  (x,c)  )

# This COV should be made from the same L in data generation and model def
COV = np.matmul(L,L.T)+ np.eye(n)*(0.01*tau)**2   # covariance matrix
w_ = np.ones(n)/10     # JK check is this correct?   p.29 'e denotes the vector of all ones' - does this answer it?
gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )

pkl.dump(  ( (n_samples,n,p,tau,deg,B,L,f,COV,gamma) , pairs ), open('portfolio_data.pkl','wb') )
