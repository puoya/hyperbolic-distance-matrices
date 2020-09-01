# Section 4.2: Weighted Tree Embedding

## Code

### Parameters

```console
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 2
        self.n_del = 1125
        self.n_del_list = 0
        self.delta = 0.9
        self.bipartite = False
        self.std = 0
        self.maxIter = 100
        self.n_del_init = 0
        self.path = '../results/weighted_tree_embedding/' 
        self.solver = ''
        self.space = 'Hyperbolic' # 'Hyperbolic', 'Euclidean'
        self.experiment = 'odor_embdding' # 'tree', 'missing_measurements' 'sensitivity'
        self.load = False
        self.cost = 'TRACE' # 'TRACE', 'LOG-DET'
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
```
- `self.N`: The number of points (52 mono-odor components)
- `self.d`: The embedding dimension
- `self.p`: The upper bound on percentage `p` of measured outliers
- `self.eps`: A minimum distance threshold to avoid trivial embeddings
- `self.K`: The number of measurements per variables
- `self.space`: The embedding space: `Hyperbolic` or `Euclidean` 
- `self.path`: The results are saved in this directory
