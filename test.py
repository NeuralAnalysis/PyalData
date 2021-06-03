import pandas as pd
import numpy as np
from importlib import reload
import os
import matplotlib.pyplot as plt

import pyaldata
reload(pyaldata)


data_dir = "/data/Mihili/"
fname = os.path.join(data_dir, "Mihili_CO_VR_2014-03-03.mat")
df = pyaldata.mat2dataframe(fname, shift_idx_fields=True)
df_ = pyaldata.restrict_to_interval(df, start_point_name='idx_movement_on', rel_start=0, rel_end=30, warn_per_trial=True)