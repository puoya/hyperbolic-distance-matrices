# Section 4.1: Missing Measurements

## Summary
Summary.


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
- `self.p`: The upper bound on percentage `p` of measured outliers
- `self.eps`: A minimum distance threshold to avoid trivial embeddings
- `self.K`: The number of measurements per variables
- `self.space`: The embedding space: `Hyperbolic` or `Euclidean` 
- `self.path`: The results are saved in this directory
