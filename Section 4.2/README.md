# Section 4.2: Weighted Tree Embedding

## Code

### Parameters

```console
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 2
        self.path = '../results/tree/'
        self.space = 'Euclidean' # 'Hyperbolic', 'Euclidean'
        self.cost = 'LOG-DET' 
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
```
- `self.N`: The number of points (ndoes of random tress)
- `self.d`: The embedding dimension
- `self.space`: The embedding space: `Hyperbolic` or `Euclidean`
- `self.cost`: The LOG-DET cost function (for hyperbolic embedding only)
- `self.norm`: The choice of matrix norm to define relative errors
- `self.error_list`: A list of increasing tolerable errors to find an admissible Gramian (denoted by \epsilon in the paper)
- `self.dela_list`: A list of descreasing regularizer weights (for log-det cost function only) 
