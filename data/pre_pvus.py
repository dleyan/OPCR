import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm
from tsl.ops.similarities import geographical_distance


total_data = []
start_time = time.strptime('2006-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_time = time.strptime('2006-12-31 23:55:00', '%Y-%m-%d %H:%M:%S')
time_length = (time.mktime(end_time) - time.mktime(start_time)) / (60*5) + 1

dir_list = os.listdir("./pvus/pvus/")
for dir in tqdm(dir_list, total=len(dir_list)):
    fn_list = os.listdir("./pvus/pvus/"+dir+"/")
    for fn in fn_list:
        if ('.csv' in fn) and ('Actual' in fn):
            df = pd.read_csv("./pvus/pvus/"+dir+"/"+fn)
            if len(df) != 0:
                temp = np.full((int(time_length)), fill_value=np.nan)
                df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%m/%d/%y %H:%M')
                df['LocalTime'] = (df['LocalTime']).astype('int64') / 1e9
                df['LocalTime'] = (df['LocalTime'] - time.mktime(start_time)) / (60*5)
                df['LocalTime'] = df['LocalTime'] - 96
                df['LocalTime'] = df['LocalTime'].astype('int64')

                temp[df['LocalTime'].values] = df['Power(MW)'].values
                total_data.append(temp[np.newaxis, ])

data = np.concatenate(total_data, axis=0)
print(data.shape)
np.save('./pvus/pvus.npy', data)

# node position 
positions = []

dir_list = os.listdir("./pvus/pvus/")
for dir in tqdm(dir_list, total=len(dir_list)):
    fn_list = os.listdir("./pvus/pvus/"+dir+"/")
    for fn in fn_list:
        if ('.csv' in fn) and ('Actual' in fn):
            df = pd.read_csv("./pvus/pvus/"+dir+"/"+fn)
            if len(df) != 0:
                lat = float(fn.split('_')[1])
                lon = float(fn.split('_')[2][1:])
                positions.append([lat, lon])

np.save('./pvus/positions.npy', np.array(positions))

# node distance
positions = np.load("./pvus/positions.npy")

dist = geographical_distance(positions, to_rad=True)
np.save("./pvus/distance.npy", dist)
