# Importing the required libraries
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import tensorflow as tf
from tabulate import tabulate
import dask
import os
import functools as ft
import glob

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def myround(x, prec=2, base=.025):
  return round(base * round(float(x)/base),prec)

def add_date_dim(xda):
    filename = os.path.basename(xda.encoding["source"])
    #Remove extension from file name
    #xda = xda.drop_dims("band")
    date_value = os.path.splitext(filename)[0][-8:]
    xda = xda.expand_dims(date_value = [date_value])
    return xda

def get_dates_from_file_patterns(file_patterns):
    date_lists = []
    for file_pattern in file_patterns:
        date_list = get_dates_from_file_names(file_pattern)
        date_lists.append(date_list)
    res = list(ft.reduce(lambda i, j: i & j, (set(x) for x in date_lists)))
    return res

def get_dates_from_file_names(file_pattern):
    #print(file_pattern)
    #List files with the pattern given
    file_list = glob.glob(file_pattern)
    #print(file_list)
    #Extract the date from the file name and add it to a list
    date_list = []
    for file in file_list:
        date_list.append(os.path.splitext(os.path.basename(file))[0][-8:])
    #print(date_list)
    return date_list

#----------------------------------------------------------------------------------------------------------------------------------
#==========
#Reading Input variables
#==========
def read_input_files(file_patterns):
    df = xr.open_mfdataset(file_patterns, preprocess = add_date_dim, combine='nested').drop(["band","spatial_ref"]).to_dataframe()
    df.reset_index(inplace=True)
    df["x"] = df["x"].apply(float).apply(myround)
    df["y"] = df["y"].apply(float).apply(myround)
    df.drop("band", axis=1, inplace=True)
    df.set_index(["x","y", "date_value"], inplace=True)
    return df

date_pattern='20*'
no2_file_pattern = f'Data/satellite/NO2/no2_kgm2_{date_pattern}.tiff'
no2flux_file_pattern = f'Data/NO2flux/no2flux_kgm2s_{date_pattern}.tiff'
rh_file_pattern = f'Data/weather/relative_humidity/RH_perc_{date_pattern}.tiff'
sr_file_pattern = f'Data/weather/solar_radiation/solarrad_Wm2_{date_pattern}.tiff'
temp_file_pattern = f'Data/weather/temperature/temp_K_{date_pattern}.tiff'
blh_file_pattern = f'Data/weather/boundary_layer_height/BLH_m_{date_pattern}.tiff'


no2_file_prefix = 'Data/satellite/NO2/no2_kgm2_'
no2flux_file_prefix = 'Data/NO2flux/no2flux_kgm2s_'
rh_file_prefix = 'Data/weather/relative_humidity/RH_perc_'
sr_file_prefix = 'Data/weather/solar_radiation/solarrad_Wm2_'
temp_file_prefix = 'Data/weather/temperature/temp_K_'
blh_file_prefix = 'Data/weather/boundary_layer_height/BLH_m_'

common_list_of_dates = get_dates_from_file_patterns([no2_file_pattern, no2flux_file_pattern, rh_file_pattern, sr_file_pattern, temp_file_pattern, blh_file_pattern])

no2_file_patterns = [f'{no2_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]
no2flux_file_patterns = [f'{no2flux_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]
rh_file_patterns = [f'{rh_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]
sr_file_patterns = [f'{sr_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]
temp_file_patterns = [f'{temp_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]
blh_file_patterns = [f'{blh_file_prefix}{date_value}.tiff' for date_value in common_list_of_dates]


no2 = read_input_files(no2_file_patterns)
no2.rename(columns={"band_data":"no2"}, inplace=True)
no2flux = read_input_files(no2flux_file_patterns)
no2flux.rename(columns={"band_data":"no2flux"}, inplace=True)
rh = read_input_files(rh_file_patterns)
rh.rename(columns={"band_data":"rh"}, inplace=True)
sr = read_input_files(sr_file_patterns)
sr.rename(columns={"band_data":"sr"}, inplace=True)
temp = read_input_files(temp_file_patterns)
temp.rename(columns={"band_data":"temp"}, inplace=True)
blh = read_input_files(blh_file_patterns)
blh.rename(columns={"band_data":"blh"}, inplace=True)

#----------------------------------------------------------------------------------------------------------------------------------

#==========
#reading GroundTruth
#==========
#Read csv file
te = pd.read_csv('Data/Taiwan_nox_emissions.csv')
tp = pd.read_csv('Data/Taiwan_powerplants.csv')

#Extract the coordinates and round them to the nearest 0.025
tp[["x","y"]] = tp["geom"].str.extract('POINT\(([0-9.]+)\s([0-9.]+)\)')
tp["x"] = tp["x"].apply(float).apply(myround)
tp["y"] = tp["y"].apply(float).apply(myround)

#Merge te and tp on facility_id
gt = pd.merge(te, tp, on='facility_id', how='left')
gt.drop(["facility_id", "name","iso2","geom","data_source_x","data_source_y","poll","unit"],axis=1,inplace=True)
gt['datetime'] = gt['datetime'].str.replace('-','')
gt.rename(columns = {"datetime":"date_value", "value" : "ground_truth_value"}, inplace=True)
gt.set_index(['x','y','date_value'], inplace=True)

#----------------------------------------------------------------------------------------------------------------------------------

#==========
#Merge input and ground truth into a single dataframe
#==========
dfs = [no2, no2flux, rh, sr, temp, blh]

input_var_join = no2.join(no2flux, on = ["x","y", "date_value"], how='inner')
input_var_join = input_var_join.join(rh, on = ["x","y", "date_value"], how='inner')
input_var_join = input_var_join.join(sr, on = ["x","y", "date_value"], how='inner')
input_var_join = input_var_join.join(temp, on = ["x","y", "date_value"], how='inner')
input_var_join = input_var_join.join(blh, on = ["x","y", "date_value"], how='inner')

merged = input_var_join.join(gt, on = ["x","y", "date_value"], how = 'inner')


#Combine the arrays into a 2d array
#model_data = np.column_stack((no2.data.flatten(),no2flux.data.flatten(),rh.data.flatten(),sr.data.flatten(),temp.data.flatten()))
#headers = ["no2", "no2flux", "rh","sr","temp"]
#table = tabulate(model_data, headers, tablefmt="fancy_grid")
#print(model_data.shape)
#print(no2)

#Remove elements from model_data if one of the 5 values is NaN
model_data = merged[~np.isnan(merged).any(axis=1)]
print(model_data.shape)


#70%, 20%, 10%
ds_train, ds_validate, ds_test = np.split(model_data.sample(frac=1, random_state=42),  [int(.7*len(model_data)), int(.9*len(model_data))])
x_train = ds_train[["no2","no2flux","rh","sr","temp", "blh"]]
x_train = ds_train[["no2flux","rh","sr","temp", "blh"]]
y_train = ds_train["ground_truth_value"]
x_validate = ds_validate[["no2","no2flux","rh","sr","temp", "blh"]]
y_validate = ds_validate["ground_truth_value"]
x_test = ds_test[["no2","no2flux","rh","sr","temp", "blh"]]
y_test = ds_test["ground_truth_value"]

x_train_min_max_scaled = x_train.copy()
for column in x_train_min_max_scaled.columns: 
    print(column)
    x_train_min_max_scaled[column] = (x_train_min_max_scaled[column] - x_train_min_max_scaled[column].min()) / (x_train_min_max_scaled[column].max() - x_train_min_max_scaled[column].min())     
  
# view normalized data 
print(df_min_max_scaled)

model = Sequential()
model.add(Dense(160, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation = 'relu'))

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train_min_max_scaled, y_train, epochs=100)
test_loss, test_acc = model.evaluate(x_validate,  y_validate, verbose=2)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1),
])

predictions = model(ds_train[:-1].numpy())



#convert model_data to tensor
ds = tf.convert_to_tensor(model_data, np.float32)
#Shuffle the tensor
ds = tf.random.shuffle(ds, seed = 5)
#Split the tensor into train, validation, and test sets

ds_train, ds_test, ds_val = tf.split(ds, [int(len(ds)*0.8), int(len(ds)*0.1), len(ds) - int(len(ds)*0.8) - int(len(ds)*0.1)])
x_train = ds_train[:0-4]
#Not using the wind since it is already incorporated into the Flux

model.fit(ds_train, y_train, epochs=5)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_validate,
)





#Build a neural network with Tensorflow using the above 5 variables

predictions = 

# View raster properties
#plot_raster(satno2, 'NO$_2$')
