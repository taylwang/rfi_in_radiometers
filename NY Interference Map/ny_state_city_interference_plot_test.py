import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

#Set Parameters

'''
year_list = [2018,2019,2020,2021,2022,2023,2024]
'''

'''
freq_list = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]
'''

year_list = [2018,2019,2020,2021,2022,2023,2024]
freq_list = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000]
'''
rad_list = ['albany', 'belleville', 'buffalo', 'chazy', 'clymer', 'jordan', 'owego', 'red_hook', 'suffern', 'tupper_lake', 'webster', 'bronx', 'east_hampton', 'queens', 'staten_island', 'stony_brook', 'wantagh']
'''
rad_list = ['albany', 'belleville', 'buffalo', 'chazy', 'clymer', 'owego', 'red_hook', 'suffern', 'tupper_lake', 'bronx', 'queens', 'staten_island', 'stony_brook', 'wantagh']
BASE_DIR = '/Volumes/500GB_Temp_Storage/NYSM_Data'
state_map = cv2.imread('/Users/taylorwang/Desktop/zussman_research/ny_state_interference_map/helper_resources/ny_state_map.jpg')
city_map = cv2.imread('/Users/taylorwang/Desktop/zussman_research/ny_state_interference_map/helper_resources/nyc_map.png')

num_year = len(year_list)
num_freq = len(freq_list)
num_rad = len(rad_list)

#--------- Helper Definitions ---------
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

#--------- Pipeline Definitions ---------

def get_rad_idx(ref_x_idx, ref_y_idx, ref_long_coord, ref_lat_coord, center_x_idx, center_y_idx, center_long_coord, center_lat_coord, rad_long_coords, rad_lat_coords, x_correction, y_correction):

	long_coord_to_idx = x_correction * (ref_x_idx - center_x_idx) / (ref_long_coord - center_long_coord)
	lat_coord_to_idx = y_correction * (ref_y_idx - center_y_idx) / (ref_lat_coord - center_lat_coord)

	rad_long_coords = np.array(rad_long_coords)
	rad_lat_coords = np.array(rad_lat_coords)

	rad_long_coords_diff = rad_long_coords - center_long_coord
	rad_lat_coords_diff = rad_lat_coords - center_lat_coord

	rad_x_idx_diff = rad_long_coords_diff * long_coord_to_idx
	rad_y_idx_diff = rad_lat_coords_diff * lat_coord_to_idx

	rad_x_idx = center_x_idx + rad_x_idx_diff
	rad_y_idx = center_y_idx + rad_y_idx_diff

	rad_x_idx = rad_x_idx.astype(int)
	rad_y_idx = rad_y_idx.astype(int)

	return rad_x_idx, rad_y_idx

def calculate_interference_instance(data):
	num_interference = 0
	for i in range(len(data)):
		data_val = data[i]
		if data_val < 0 or data_val > 350:
			num_interference += 1

	return num_interference

#Generate a list containing the interference counts
#The interfernce count uses day binning: If a day has interference, we consider that one count of interference
#The list structure is radiometer location -> frequency -> year
#The lists are stored in pickle files
#folder is the folder containing all the pickle files
def load_interference_list(folder):

	#We could technically reduce the time complexity of this function by a factor of N
	#However, loading the data like this makes the for loops easy to follow
	#We prefer to have it like this so we can easily work off it later

	folder = Path(folder)
	paths = [p for p in folder.iterdir() if p.is_file() and p.suffix == '.pkl']

	rad_list = []
	for path in paths:
		with open(path, 'rb') as f:
			location_inter_list = pickle.load(f)
			rad_list.append(location_inter_list)

	rad_inter_list = []
	for freq_list in rad_list: 
		freq_inter_list = []
		for year_list in freq_list:
			year_inter_list = []
			for day_list in year_list:
				year_inter_count = 0
				for day in day_list:
					day_inter_count = day[0]
					if(day_inter_count != 0):
						year_inter_count += 1
				year_inter_list.append(year_inter_count)
			freq_inter_list.append(year_inter_list)
		rad_inter_list.append(freq_inter_list)

	return rad_inter_list 

def get_interference_list(year_list, freq_list, rad_list):

	#Generate a list containing the interference counts
	#Each data point indicating interfernce adds to the interference count. There is no day binning
	#The list structure goes radiometer location -> frequency -> year
	
	'''
	year_inter = []
	for i in range(num_year):
		freq_inter = []
		for j in range(num_freq):
			rad_inter = []
			for k in range(num_rad):
				bright_dates, bright_vals = load_brightness_data(rad_list[k], year_list[i], 'lv1', freq_list[j], 'brightness')
				interference_instance = calculate_interference_instance(bright_vals)

				rad_inter.append(interference_instance)
			
			freq_inter.append(rad_inter)	
		year_inter.append(freq_inter)

	return year_inter

	'''

	rad_inter = []
	for i in range(num_rad):
		freq_inter = []
		for j in range(num_freq):
			year_inter = []
			for k in range(num_year):
				bright_dates, bright_vals = load_brightness_data(rad_list[i], year_list[k], 'lv1', freq_list[j], 'brightness')
				interference_instance = calculate_interference_instance(bright_vals)

				year_inter.append(interference_instance)
			freq_inter.append(rad_inter)	
		rad_inter.append(freq_inter)

	return rad_inter

#--------- Main --------- 

#Find the indices of the radiometer for the new york state and new york city maps we use

#State
state_ref_x_idx = 729
state_ref_y_idx = 101

state_ref_long_coord = -73.34491
state_ref_lat_coord = 45.01065

state_center_x_idx = 597
state_center_y_idx = 417

state_center_long_coord = -75.24790
state_center_lat_coord = 42.86919

state_x_correction = 1.55
state_y_correction = 1

'''
state_rad_names = ['Albany', 'Belleville', 'Buffalo', 'Chazy', 'Clymer', 'Jordan', 'Owego', 'Red Hook', 'Suffern', 'Tupper Lake', 'Webster', 'Bronx', 'East Hampton', 'Queens', 'Staten Island', 'Stony Brook', 'Wantagh']
'''

state_rad_names = ['Albany', 'Belleville', 'Buffalo', 'Chazy', 'Clymer', 'Owego', 'Red hook', 'Suffern', 'Tupper Lake', 'Bronx', 'Queens', 'Staten Island', 'Stony_brook', 'Wantagh']

state_rad_long_coords = [-73.81128, -76.11765, -78.79461, -73.46634, -79.69746, -76.25307, -73.88412, -74.08597, -74.44105, -73.89352, -73.81585, -74.14849, -73.13328, -73.5054]
state_rad_lat_coords = [42.75175, 43.78823, 42.99359, 44.889, 42.12143, 42.02493, 41.99983, 41.13303, 44.22425, 40.87248, 40.73433, 40.65401, 40.91957, 40.65025]

'''
state_rad_long_coords = [-73.81128, -76.11765, -78.79461, -73.46634, -79.69746, -76.46999, -76.25307, -73.88412, -74.08597, -74.44105, -77.41238, -73.89352, -72.20094, -73.81585, -74.14849, -73.13328, -73.5054]
state_rad_lat_coords = [42.75175, 43.78823, 42.99359, 44.889, 42.12143, 43.06874, 42.02493, 41.99983, 41.13303, 44.22425, 43.2601, 40.87248, 40.97039, 40.73433, 40.65401, 40.91957, 40.65025]
'''

state_rad_x_indices, state_rad_y_indices = get_rad_idx(state_ref_x_idx, state_ref_y_idx, state_ref_long_coord, state_ref_lat_coord, state_center_x_idx, state_center_y_idx, state_center_long_coord, state_center_lat_coord, state_rad_long_coords, state_rad_lat_coords, state_x_correction, state_y_correction)

#City
city_ref_x_idx = 1454
city_ref_y_idx = 276

city_ref_long_coord = -71.86195
city_ref_lat_coord = 41.07189

city_center_x_idx = 795
city_center_y_idx = 367

city_center_long_coord = -73.15311
city_center_lat_coord = 40.96498

city_x_correction = 1
city_y_correction = 1

city_rad_names = ['Bronx', 'Queens', 'Staten Island', 'Stony Brook', 'Wantagh']
city_rad_long_coords = [-73.89352, -73.81585, -74.14849, -73.13328, -73.5054]
city_rad_lat_coords = [40.87248, 40.73433, 40.65401, 40.91957, 40.65025]

'''
city_rad_long_coords = [-73.89352, -72.20094, -73.81585, -74.14849, -73.13328, -73.5054]
city_rad_lat_coords = [40.87248, 40.97039, 40.73433, 40.65401, 40.91957, 40.65025]
'''

city_rad_x_indices, city_rad_y_indices = get_rad_idx(city_ref_x_idx, city_ref_y_idx, city_ref_long_coord, city_ref_lat_coord, city_center_x_idx, city_center_y_idx, city_center_long_coord, city_center_lat_coord, city_rad_long_coords, city_rad_lat_coords, city_x_correction, city_y_correction)

#Obtain the complete interference list for each year, frequency, and radiometer
'''
year_freq_rad_inter_list = get_interference_list(year_list, freq_list, rad_list)
'''

rad_freq_year_inter_list = load_interference_list('/Users/taylorwang/Desktop/zussman_research/rfi_complete_array/saved_rfi_arrays')

#Take the natural log of the interference list, then rescale to range from 0 to 255 for display purposes
disp_inter_list = rad_freq_year_inter_list.copy()
disp_inter_list = np.array(disp_inter_list)
disp_inter_list = disp_inter_list ** (1/2)
disp_inter_list[disp_inter_list < 0] = 0
max_inter = np.max(disp_inter_list)
disp_inter_list = disp_inter_list / max_inter * 255

#Obtain indices to place the radiometer name by shifting slightly from the radiometer position. Some specific radiometers may need special treatment to not overlap with the others.
state_rad_text_x_indices = state_rad_x_indices.copy()
state_rad_text_y_indices = state_rad_y_indices.copy()
state_rad_text_x_indices = state_rad_x_indices + 15
state_rad_text_y_indices[5] = state_rad_text_y_indices[5] + 10

city_rad_text_x_indices = city_rad_x_indices.copy()
city_rad_text_y_indices = city_rad_y_indices.copy()
city_rad_text_x_indices = city_rad_x_indices + 40
'''
city_rad_text_x_indices[3] = city_rad_x_indices[3] - 100
city_rad_text_y_indices[3] = city_rad_y_indices[3] + 80
city_rad_text_x_indices[1] = city_rad_x_indices[1] - 100
city_rad_text_y_indices[1] = city_rad_y_indices[1] - 50
'''

'''
comb_rad_names = state_rad_names + city_rad_names

comb_rad_x_indices = np.concatenate((state_rad_x_indices, city_rad_x_indices))
comb_rad_y_indices = np.concatenate((state_rad_y_indices, city_rad_y_indices))
'''

for i in range(num_year):
	for j in range(num_freq):
		
		new_state_map = state_map.copy()
		new_state_map = np.array(new_state_map)
		new_city_map = city_map.copy()
		new_city_map = np.array(new_city_map)

		for k in range(num_rad):

			#Draw circles on all radiometer locations on the state map. The circle color depends on the interference value.
			disp_inter_val = disp_inter_list[k][j][i]
			real_inter_val = rad_freq_year_inter_list[k][j][i]

			rad_x_idx = state_rad_x_indices[k]
			rad_y_idx = state_rad_y_indices[k]

			center_coords = (rad_x_idx, rad_y_idx)
			radius = 10
			print('this is disp_inter_val')
			print(disp_inter_val)
			print('')
			color = (disp_inter_val, 0, 0)
			thickness = -1
			cv2.circle(new_state_map, center_coords, radius, color, thickness)

			if state_rad_names[k] in city_rad_names:
				
				#Draw circles on the radiometer locations on the city map. The circle color depends on the interference value.
				rad_idx = city_rad_names.index(state_rad_names[k])
				city_rad_x_idx = city_rad_x_indices[rad_idx]
				city_rad_y_idx = city_rad_y_indices[rad_idx]

				center_coords = (city_rad_x_idx, city_rad_y_idx)
				radius = 20
				color = (disp_inter_val, 0, 0)
				thickness = -1
				cv2.circle(new_city_map, center_coords, radius, color, thickness)

				#Label the radiometer locations on the city map.
				text = state_rad_names[k] + ":" + str(real_inter_val)
				position = (city_rad_text_x_indices[rad_idx], city_rad_text_y_indices[rad_idx])
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = 1.6
				color = (0, 0, 0)
				thickness = 4              

				#Draw a white background rectangle behind each radiometer label so that its easier to read
				(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
				top_left = (position[0] - 10, position[1] - text_height - baseline)
				bottom_right = (position[0] + text_width + 10, position[1] + baseline)
				cv2.rectangle(new_city_map, top_left, bottom_right, (255,255,255), thickness=cv2.FILLED)

				#Write the label after the rectangle is drawn
				cv2.putText(new_city_map, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

				continue

			else:

				state_rad_x_idx = state_rad_text_x_indices[k]
				state_rad_y_idx = state_rad_text_y_indices[k]

				text = state_rad_names[k] + ":" + str(real_inter_val)
				position = (state_rad_x_idx, state_rad_y_idx)
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = 1
				color = (0, 0, 0)
				thickness = 2                
				cv2.putText(new_state_map, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

		fig, axes = plt.subplots(
		    2, 1, 
		    figsize=(4, 5), 
		    gridspec_kw={'height_ratios': [1.64, 1]}  # first plot twice as tall as second
		)

		axes[0].set_xticks([])
		axes[0].set_yticks([])

		axes[1].set_xticks([])
		axes[1].set_yticks([])

		axes[0].imshow(new_state_map)
		axes[1].imshow(new_city_map)

		title = f"Year: {year_list[i]}   Frequency: {freq_list[j]} GHz"
		fig.suptitle(title, fontsize=12, y=0.92)      # <<--- FIGURE TITLE	
		plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for title


		name = f"{year_list[i]}_{freq_list[j]}.png"
		folder = '/Users/taylorwang/Desktop/zussman_research/ny_state_interference_map/data/day_binning_above_350k'
		path = os.path.join(folder,name)

		plt.savefig(path, dpi=300)
		plt.close()

		print('saved')

			
