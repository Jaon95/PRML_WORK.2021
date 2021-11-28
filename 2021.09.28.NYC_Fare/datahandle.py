import  numpy as  np
import  pandas as pd
import  matplotlib.pyplot as  plt
import  seaborn as sns
import  os

plt.style.use('seaborn-whitegrid')

df_train = pd.read_csv(os.getcwd()+'/2021.09.28.NYC_Fare/train_raw.csv',parse_dates=["pickup_datetime"])

print(df_train.head())

print(df_train.dtypes)

print(df_train.describe())

df_train = df_train[df_train.fare_amount>=0]

df_train[df_train.fare_amount<100].fare_amount.hist(bins=100,figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histgram')

print(df_train.isnull().sum())

print('old size : %d' % len(df_train))

df_train.dropna(how='any',axis='rows')

def select_within_boundingbox(df,BB):
    return  (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

BB = (-74.5, -72.8, 40.5, 41.8)
nyc_map = plt.imread('2021.09.28.NYC_Fare/nyc_-74.5_-72.8_40.5_41.8.png')

BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = plt.imread('2021.09.28.NYC_Fare/nyc_-74.3_-73.7_40.5_40.9.png')

print('old size :%d '% len(df_train))
df_train = df_train[select_within_boundingbox(df_train,BB)]

def plot_on_map(df,BB,nyc_map,s=10,alpha = 0.2):
    fig,axs = plt.subplot(1,2,figsize=(16,10))
