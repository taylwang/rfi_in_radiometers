import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
from collections import defaultdict


# === CONFIGURATION ===
'''
SITES = ['Albany', 'Belleville', 'Buffalo', 'Chazy', 'Clymer', 'Jordan', 'Owego', 'Red_hook', 'Suffern', 'Tupper_lake', 'Webster', 'East_Hampton', 'Queens', 'Staten_Island', 'Stony_brook', 'Wantagh']
'''
SITES = ['Queens', 'Staten_Island', 'Stony_brook', 'Wantagh']
YEARS = [2018,2019,2020,2021,2022,2023,2024]  
FREQS = [22.5,22.234,23.034,23.834,25.0,26.234,28.0,30.0]
'''
YEARS = [2018,2019,2020,2021,2022]
FREQ = [22.5, 22.234, 23.034, 23.834,25.0,26.234,28.0,30.0]
'''
BASE_DIR = "/Volumes/500GB_Temp_Storage/NYSM_Data"
OUTPUT_DIR = f"{SITES[0]}_{YEARS[0]}_{FREQS[0]}"
FIGSIZE = (14, 4)

def load_data(site, year, level, key, subkey):
    prefix = f"{key}_{subkey}" if level != "lv2" else key
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)
    return pd.to_datetime(dates), values

def align_times(smaller_dates, smaller_vals, bigger_dates, bigger_vals):

    smaller_dates = np.array(smaller_dates)
    smaller_vals = np.array(smaller_vals)
    bigger_dates = np.array(bigger_dates)
    bigger_vals = np.array(bigger_vals)

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

    aligned_dates = np.array(aligned_dates)
    aligned_vals = np.array(aligned_vals)

    return aligned_dates, aligned_vals

# === MAIN LOOP with sub-year ranges ===

for site in SITES:
    freq_list = []
    for freq in FREQS:
        year_list = []
        for year in YEARS:

            #Keep only the date in bt_times. Then convert it to a numpy string array for ease of use.
            VOLT_TIMES, VOLT_VALUES = load_data(site, year, "lv0", freq, 'voltages')
            BT_TIMES, BT_VALUES = load_data(site, year, "lv1", freq, "brightness")
            VAPOR_TIMES, VAPOR_VALUES = load_data(site, year, "lv2", "Int. Vapor(cm)", "values")

            bt_dates = BT_TIMES.date
            bt_dates = np.array(bt_dates)
            bt_dates = bt_dates.astype(str)

            DAYS = np.unique(bt_dates)

            #Shorten each data array to the shortest length out of all of them, and align the times
            volt_array_len = len(VOLT_TIMES)
            bt_array_len = len(BT_TIMES)
            vapor_array_len = len(VAPOR_TIMES)

            data_time_array = [VOLT_TIMES, BT_TIMES, VAPOR_TIMES]
            data_value_array = [VOLT_VALUES, BT_VALUES, VAPOR_VALUES]
            length_array = [volt_array_len, bt_array_len, vapor_array_len]

            min_length = min(length_array)
            min_index = length_array.index(min_length)

            for i in range(len(length_array)):
                if i != min_index:
                    data_time_array[i], data_value_array[i] = align_times(data_time_array[min_index], data_value_array[min_index], data_time_array[i], data_value_array[i])

            VOLT_TIMES = data_time_array[0]
            BT_TIMES = data_time_array[1]
            VAPOR_TIMES = data_time_array[2]

            VOLT_VALUES = data_value_array[0]
            BT_VALUES = data_value_array[1]
            VAPOR_VALUES = data_value_array[2]
            #End of array shortening scheme

            day_list = []
            for day in DAYS:

                day_indices = np.where(bt_dates == day)[0]

                volt_values = VOLT_VALUES[day_indices]
                bt_values = BT_VALUES[day_indices]
                vapor_values = VAPOR_VALUES[day_indices]

                rfi_instances = 0
                for bt_value in bt_values:
                    if bt_value > 350:
                        rfi_instances += 1

                data_list = [rfi_instances, volt_values, bt_values, vapor_values]
                
                bright_temp_values = data_list[2]
                radio_freq_instance = data_list[0]

                day_list.append(data_list)

            year_list.append(day_list)
        freq_list.append(year_list)

    complete_list = freq_list

    with open(f"/Users/taylorwang/Desktop/zussman_research/rfi_complete_array/saved_rfi_arrays/{site}.pkl", "wb") as f:
        pickle.dump(complete_list, f)
