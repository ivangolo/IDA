import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
X = csr_matrix(mmread('test.x.nm'))
y = np.loadtxt('test.y.dat')
