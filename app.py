import numpy as np
import pandas as pd
import math
import csv

class N:
    def __init__(self, a): self.attribute, self.children, self.answer = a, {}, ""

# Tree Construction (C) - Main Logic
def C(d, m):
    # Embedded Entropy Lambda (H)
    H = lambda s: -(pd.Series(s).value_counts(normalize=True).pipe(lambda v: (v * np.log2(v)).sum())) if len(np.unique(s)) > 1 else 0
    # Embedded Gain Ratio Calculation (R)
    def R(col_idx):
        t, f, g, s = d[:, -1], d[:, col_idx], H(d[:, -1]), 0
        for v in np.unique(f):
            sub = t[f == v]
            p = len(sub) / len(t)
            g -= p * H(sub)
            s -= p * math.log2(p)
        return g / s if s != 0 else 0

    t = d[:, -1]
    if t.size > 0 and (np.unique(t)).size == 1:
        n = N(""); n.answer = t[0]; return n
    i = np.argmax([R(c) for c in range(d.shape[1] - 1)])
    n = N(m[i])
    m, f = np.delete(m, i, 0), d[:, i]
    
    for v in np.unique(f):
        n.children[v] = C(np.delete(d[f == v], i, 1), m)
    return n

# Tree Printing (P)
I = lambda l: "   " * l
def P(n, l=0):
    if n.answer != "": print(f"{I(l)}--> {n.answer}"); return
    print(f"{I(l)}{n.attribute}?")
    for v, c in n.children.items():
        print(f"{I(l)}   [{v}]")
        P(c, l + 1)

# Execution
if __name__ == "__main__":
    # Data Reading (D) is now a single line in execution
    m, d = list((df := pd.read_csv("tennisdata.csv")).columns), df.to_numpy(dtype=str)
    P(C(d, m))
