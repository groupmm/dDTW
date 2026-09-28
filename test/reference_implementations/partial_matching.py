# computes partial matching. Code taken from the FMP notebooks https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html
# The corresponding notebook was created by Meinard Müller, Tim Zunner, and Frank Zalkow. 

import numpy as np
from numba import jit

@jit(nopython=True)
def compute_partial_matching(S):
    """Given the score matrix, compute the accumulated score matrix
    for partial matching

    Notebook: C7/C7S3_CommonSubsequence.ipynb

    Args:
        S (np.ndarray): Score matrix

    Returns:
        D (np.ndarray): Accumulated score matrix
        P (np.ndarray): Partial match (array of index pairs)
    """
    N, M = S.shape
    D = np.zeros((N+1, M+1))
    for n in range(1, N+1):
        for m in range(1, M+1):
            D[n, m] = max(D[n, m-1], D[n-1, m], D[n-1, m-1] + S[n-1, m-1])

    P = []
    n = N
    m = M
    while (n > 0) and (m > 0):
        if D[n, m] == D[n, m-1]:
            m = m - 1
        elif D[n, m] == D[n-1, m]:
            n = n - 1
        else:
            P.append((n-1, m-1))
            n = n - 1
            m = m - 1
    P.reverse()
    P = np.array(P)
    return D, P