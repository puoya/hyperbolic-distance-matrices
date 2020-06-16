import numpy as np
import site
site.addsitedir('../lib/') 
import htools
import hdm
class parameters:
    def __init__(self):
        self.N = 52
        self.d = 2
        self.p = 0.1
        self.eps = 0.25
        self.K = .4
        self.mode = 'Euclidean' # or Euclidean
        self.path = 'results/'
param = parameters()
# Load biochemical (odor) component measurements of bluberries in a matrix form
R = np.load('data/biochemical_components.npy')
# Calculate corss-correlation matrix for odor components
C = htools.cross_correlation(R)
# Calculate embedding
output = hdm.odor_embedding(C,param)
# Print main results
print(param.mode,'embedding accuracy:', output.r)
print('Embedding dimension:', param.d)
x = output.x
outliers = len(x[x>param.eps])
print('Number of outliers:', outliers)
# Save results and parameters
np.save(param.path+'output',output)
np.save(param.path+'param',param)
X = htools.d2x(output.D,param)
np.save(param.path+'X',X)
