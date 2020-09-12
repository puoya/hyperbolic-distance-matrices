# Section 4.3: Odor Embedding

## Code

### Parameters

```console
class parameters:
    def __init__(self):
        self.N = 52
        self.d = 2
        self.p = 0.005
        self.eps = 0.25
        self.K = 4
        self.space = 'Hyperbolic'
        self.path = '/results/'
param = parameters()
```
- `self.N`: The number of points (52 mono-odor components)
- `self.d`: The embedding dimension
- `self.p`: The upper bound on percentage `p` of allowable outliers
- `self.eps`: A minimum distance constraints to avoid trivial embeddings
- `self.K`: The number of measurements (binray comparisons) per variables
- `self.space`: The embedding space: `Hyperbolic` or `Euclidean` 
- `self.path`: The experiment results are saved in this directory
