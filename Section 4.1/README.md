# Section 4.1: Missing Measurements

## Code

### Parameters

```console
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 4
        self.n_del = 0
        self.delta = 0.01
        self.maxIter = 100
        self.n_del_init = 0
        self.path = '../results/missing_measurements/' #'../results/sensitivity/'
        self.experiment = 'missing_measurements' # 'sensitivity'
        self.cost = 'TRACE' # 'TRACE', 'LOG-DET'
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
```
- `self.N`: The number of points (ndoes of random tress)
- `self.d`: The embedding dimension
- `self.n_del_init`: The initial number of missing metric measurement
- `self.n_del`: The number of missing metric measurement (at each iteration)
- `self.maxIter`: The maximum number of iterations (denoted by M in the paper)
- `self.cost`: The "Trace" cost function (for hyperbolic and Euclidean embeddings)
- `self.norm`: The choice of matrix norm to define relative errors
- `self.error_list`: A list of increasing tolerable errors to find an admissible Gramian (denoted by \epsilon in the paper)
- `self.delta_list`: A list of descreasing regularizer weights  
