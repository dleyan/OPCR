import numpy as np
import pandas as pd
import datetime
from tsl.ops.similarities import correntropy
from tqdm import tqdm


df_list = []
for i in range(6):

    df = pd.read_csv(f"./cer/File{str(i+1)}.txt", sep=' ')
    df.columns = ['id', 'time', 'data']
    df_list.append(df)

total_df = pd.concat(df_list)

# 2009-07-15 00:00:00
start_t = total_df['time'].min()
end_t = total_df['time'].max()
time_length = (end_t//100 - start_t//100) * 48 + (end_t%100)

start_time = datetime.datetime.strptime('2009-01-01 00:00:00', '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=int(start_t//100), minutes=int(start_t%100-1)*30)
print("start time: ", start_time)
print("time length: ", time_length)


data_list = []
group = total_df.groupby('id')
for id, df_g in tqdm(group, total=len(group)):

    df_g.sort_values(by='time', ascending=True, inplace=True)

    temp = np.full((time_length), fill_value=np.nan)

    times = df_g['time'].values
    times = (times//100 - start_t//100) * 48 + (times%100) - 1
    times = times.astype(np.int)

    temp[times] = df_g['data'].values

    data_list.append(temp[np.newaxis, ])

print(np.concatenate(data_list, axis=0).shape)
np.save("./cer/cer.npy", np.concatenate(data_list, axis=0))

# 2009-07-15 00:00:00
# 2010-01-01
# (6435, 25728)
# 31+30+31+30+31+17
data = np.load("./cer/cer.npy")

drop_len = (31+30+31+30+31+17) * 48
train_len = 120 * 48

train_data = data[:, drop_len:drop_len+train_len]

# compute similarity
train_data[np.isnan(train_data)] = 0
sims = correntropy(x=train_data.T, period=48)

np.save('./cer/distance.npy', sims)
