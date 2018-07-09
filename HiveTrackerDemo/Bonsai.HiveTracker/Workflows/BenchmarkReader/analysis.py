# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:56:18 2018

@author: gonca
"""

import numpy as np
import matplotlib.pyplot as plt

fname = r'C:/Users/gonca/Documents/Projects/github/hivetracker/bonsai-interface/BinaryReader/base0Axis0.bin'
data = np.fromfile(fname,dtype=np.float32)
data = data.reshape((-1,4))

plt.plot(data)
