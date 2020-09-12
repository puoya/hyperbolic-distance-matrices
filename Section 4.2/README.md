# Section 4.2: Weighted Tree Embedding

## Code

### Parameters

```console
class parameters:
def __init__(self):
    self.N = 10
    self.d = 2
    self.path = '../results/tree/'
    self.space = 'Hyperbolic' # 'Hyperbolic', 'Euclidean'
    self.cost = 'LOG-DET' 
    self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
    self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
    self.error_list = 10**(np.linspace(-2, 0, num=5))
    self.delta_list = 10**(np.linspace(0, -3, num=5))
```
- `self.N`: The number of points (nodes of random trees)
- `self.d`: The embedding dimension
- `self.path`: The experimental results are saved in this directory
- `self.space`: The embedding space: `Hyperbolic` or `Euclidean`
- `self.cost`: LOG-DET cost function (for hyperbolic embedding)
- `self.norm`: The choice of norm to define relative errors
- `self.K`: The number of measurements per variables
- `self.error_list`: Increasing sequence of tolerable relative error (epsilon) to estimate a valid Gramian 
- `self.delta_list`: Diminishing sequence of deltas (see specifications of log-det cost function)
