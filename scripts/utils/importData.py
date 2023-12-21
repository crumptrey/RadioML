# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
import numpy as np
import matplotlib.pyplot as plt
import random, sys
import pickle
from typing import Tuple, Dict
from collections import defaultdict
'''
Unpickles dataset and returns data and description
'''

def importData(file):
    with open(file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    description = defaultdict(list)
    # Declare j just to get the linter to stop complaining about the lamba below
    j = None
    snrs, mods = map(
        lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0]
    )
    for mod in mods:
        for snr in snrs:
            description[mod].append(snr)

    return data, description








