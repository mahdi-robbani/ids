### Understanding covariance and correlation

import numpy as np

# Sample (look at the values of sigma)
Mean = np.array([2,3])
Sigma = np.array([[6, -3], [-3, 7]])
x, y = np.random.multivariate_normal(Mean, Sigma, 5000).T


def cova(x,y):
    xdiff = []
    ydiff = []
    for i in x:
        xdiff.append((i - np.mean(x))) 
    for j in y:
        ydiff.append((j - np.mean(y)))
    xdiff = np.array(xdiff)
    ydiff = np.array(ydiff)
    cov = sum(xdiff * ydiff) / len(x)
    return cov

def pearson(x,y):                                   # pearson correlation is basically the covariance normalized by the sd
    pearson = cova(x,y) / (np.std(x)*np.std(y))
    return(pearson)

print("\nSigma =\n", Sigma)

print("\nnp.cov =\n", np.cov(x,y))
covariance = cova(x,y)
correlation = pearson(x,y)
print("\nCovariance =", covariance)
print("Std(x) =", np.var(x))
print("Std(y) =", np.var(y))

np_correlation = np.corrcoef(x,y)
print("\nnp.corrcoef =\n", np_correlation)
print("\nCorrelation =", correlation)