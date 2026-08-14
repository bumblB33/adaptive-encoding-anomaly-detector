# /usr/bin/env python3
from sklearn.neighbors import LocalOutlierFactor

LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
