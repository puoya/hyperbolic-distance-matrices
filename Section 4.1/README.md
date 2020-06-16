# KEDM

This repository provides the source codes to regenerate the results provided in "On the move: Localization with Kinetic Euclidean Distance Matrices" paper, submitted to ICASSP2019.

## Summary
We regenerate the following simulations:
- Illustration of KEDM ambiguities: ```kedm_ambiguity.py```
- Noisy measurement experiment: ```sketch_experiment.py```
- Missing distance measurements experiment:```sparsity_experiment.py``` 

## Requirements
This python code has been proved to work with the following installed.
- cvxopt==1.2.2
- cvxpy==1.0.10
- numpy==1.15.4
- python==3.7.0

## Code

### Parameters class
The fundamental step in conducting all the expertiments is setting up the necessary parameters which is as follows:

```console
class parameters:
def __init__(self):
self.N = 6
self.d = 2
self.P = 3
self.omega = 2*np.pi
self.mode = 1
self.T_trn = np.array([-1, 1])
self.T_tst = np.array([-1, 1])
self.N_trn = ktools.K_base(self)
self.K = ktools.K_base(self)
self.N_tst= 500
self.Nr = 5
self.n_del = 0
self.sampling = 1
self.delta = 0.99
self.std = 1
self.maxIter = 5
self.n_del_init = 0
self.bipartite = False
self.N0 = 3
self.Pr = 0.9
self.path = '../../../results/kedm/python3/'
```
Let us now briefly explain each parameter (for more information, please read [KEDM-ICASSP19](https://github.com/swing-research/kedm-pubs/tree/master/icassp)).
- `self.N`: Number of moving point,
- `self.d`: Embedding dimension of trajectories,
- `self.P`: Degree of polynomial model,
- `self.omega`: Base frequency of bandlimited trajectories. No need to change for polynomial model;
- `self.mode`: `1` for polynomial and `2` for bandlimited model,
- `self.T_trn`: Train interval, where we can sample measurements,
- `self.T_tst`: Test interval, where we estimate estimation error,
- `self.N_trn`: Number of temporal samples, denoted by `T` in the paper,
- `self.K`: Number of basis Gramians, `2P+1` for polynomial and `4P+1` for bandlimited model,
- `self.N_tst`: Number of test samples to approximate estimation error `e_X`,
- `self.Nr`: Number of time samples for positive semidefinite constraing `G(t_i) >> 0`,
- `self.n_del`: Number missing distances at a time,
- `self.sampling`: Sampling protocol: `1` for equi-distance, `2` for Chebyshev and `3` for random,
- `self.delta`: Successful estimation threshold,
- `self.std`: Standard deviation of measurement noise,
- `self.maxIter`: Maximum number of iterations,
- `self.n_del_init`: Number of initial missing measurements (only use for estimating sparsity level),
- `self.bipartite`: Boolean parameter: `True` for bipartite and `False` for general measurement mask
- `self.N0`: Number of points in an independent set of a bipartite measurement mask,
- `self.Pr`: Probability of successful estimation,
- `self.path`: Save the results in this directory.



## KEDM ambiguities

The script ```kedm_ambiguity.py``` gives us an illustration that explain KEDM general distance ambiguities. There are few paramters to tweak.
- `colors = np.random.rand(3,N)` specifies the colors of each trajectory,
- `fig_name` specifies the name of the produced figure,
- `A = ktools.randomAs(param)` generates random coefficients for a trajectory we wish to study.

## Noisy measurement experiment

You can regenerate the noisy measurement experiment with running the script `sketch_experiment.py`. There are some exclusive paramters to choose:
- `colors = np.random.rand(3,N)` specifies the colors of each trajectory,
- `fig_name` specifies the name of the produced figure,
- `A = ktools.randomAs(param)` generates random coefficients for a trajectory we wish to study.
- `eDi, eDo` are relative measured and estimated distance errors, and `eX` is the relative trajectory mismatch.

The function `kedm.Save(param, kedm_output, trj_output, '100')` saves the results.

## Missing distance measurements experiment

The script `sparsity_experiment.py` simply estimates the maximum sparsity. The function `kedm.FindMaxSprs(param)` returns the maximum sparsity level ```S``` and `kedm.Save(param, kedm_output, trj_output, '100')` saves the results. Note that with `self.n_del_init` you can set the minimum sparsity level to be tested.


