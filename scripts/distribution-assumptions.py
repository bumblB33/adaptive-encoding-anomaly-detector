#!/usr/bin/env python3
from sklearn.neighbors import LocalOutlierFactor

from config.defaults import LOF_CONTAMINATION

LOF = LocalOutlierFactor(contamination=LOF_CONTAMINATION, novelty=True)
