import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import sys
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

#Retrieving the brightness temperature data.
def load_brightness_data(site, year, level, key, subkey):
    prefix = f"{key}_{subkey}" if level != "lv2" else key
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)
    
    return pd.to_datetime(dates), values

#Retrieving the vapor data.
def load_vapor_data(site, year, level, key):
    prefix = f"{key}"
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)

    return pd.to_datetime(dates), values

#Settings and Constant Values
SITE = 'queens'
YEAR = '2022'
BRIGHT_LEVEL = 'lv1'
VAPOR_LEVEL = 'lv2'
BASE_DIR = "./NYS_Mesonet_Data"
OUTPUT_DIR = "RFI_detection_plots_and_csv"
FREQ_LIST = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]

bright_matrix_vals = []
bright_matrix_dates = []

for i in range(len(FREQ_LIST)):
	bright_dates, bright_vals = load_brightness_data(SITE, YEAR , BRIGHT_LEVEL, FREQ_LIST[i], 'brightness')
	bright_matrix_vals.append(bright_vals)
	bright_matrix_dates.append(bright_dates)

vapor_dates, vapor_vals = load_vapor_data(SITE, YEAR, VAPOR_LEVEL, 'Int. Vapor(cm)')

len_bright_dates = len(bright_dates)
len_vapor_dates = len(vapor_dates)

smaller_dates = np.array([])
smaller_vals = np.array([])
bigger_dates = np.array([])
bigger_vals = np.array([])

if len_bright_dates > len_vapor_dates:
    smaller_dates = vapor_dates
    smaller_vals = vapor_vals
    bigger_dates = bright_dates
    bigger_vals = bright_vals
    print('bright dates larger')

if len_bright_dates < len_vapor_dates:
    smaller_dates = bright_dates
    smaller_vals = bright_vals
    bigger_dates = vapor_dates
    bigger_vals = vapor_vals
    print('vapor_dates larger')

# Find insertion positions
positions = np.searchsorted(bigger_dates, smaller_dates)

# Get closest timestamps
aligned_dates = []
aligned_vals = []
for i, pos in enumerate(positions):
    if pos == 0:
        aligned_dates.append(bigger_dates[0])
        aligned_vals.append(bigger_vals[0])
    elif pos == len(bigger_dates):
        aligned_dates.append(bigger_dates[-1])
        aligned_vals.append(bigger_vals[-1])
    else:
        # Compare neighbors to find the closest
        prev_val = bigger_dates[pos - 1]
        next_val = bigger_dates[pos]
        if abs(smaller_dates[i] - prev_val) <= abs(smaller_dates[i] - next_val):
            aligned_dates.append(bigger_dates[pos-1])
            aligned_vals.append(bigger_vals[pos-1])
        else:
            aligned_dates.append(bigger_dates[pos])
            aligned_vals.append(bigger_vals[pos])

aligned_dates = pd.to_datetime(aligned_dates)

if len_bright_dates > len_vapor_dates:
    bright_dates = aligned_dates
    bright_vals = aligned_vals

if len_bright_dates < len_vapor_dates:
    vapor_dates = aligned_dates
    vapor_vals = aligned_vals







