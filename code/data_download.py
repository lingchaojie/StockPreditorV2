import bitmex
import os, sys, time
import config
import pandas as pd
from datetime import datetime, timedelta, timezone


client = bitmex.bitmex(test=False, api_key=config.api_key, 
                       api_secret=config.api_secret)

def time_to_str(datetime_object):
    # convert time into string representation
    return datetime_object.strftime(config.time_format)

def str_to_time(str_time):
    # convert string time into datatime object
    return datetime.strptime(str_time, config.time_format)
    
def get_data_to_df(data):
    # convert data from dictionary to pandas dataframe
    data_dict = {}
    for i in range(len(data[0])):
        dt_str = time_to_str(data[0][i]['timestamp'])
        data_dict[dt_str] = {}
        data_dict[dt_str]["Date"] = time_to_str(data[0][i]['timestamp'])
        data_dict[dt_str]["Open"] = data[0][i]["open"]

        data_dict[dt_str]["High"] = data[0][i]["high"]
        data_dict[dt_str]["Low"] = data[0][i]["low"]
        data_dict[dt_str]["Close"] = data[0][i]["close"]
        data_dict[dt_str]["Volume"] = data[0][i]["volume"]

    df = pd.DataFrame.from_dict(data_dict,orient='index')
    return df

def get_start_time():
    # try to get time from previous file, if not exist, start from default time 
    try:
        df = pd.read_csv('{}/{}.csv'.format(config.save_path, config.symbol),index_col=0)
        start_time = str_to_time(df.index[-1]) + timedelta(minutes=1)
        print('previous file found, continue downloading, {}'.format(start_time))
        exist = 1
    except:
        df = 0
        start_time = str_to_time(config.default_start_time)
        print('previous file not found, downloading from default start time, {}'.format(config.default_start_time))
        exist = 0
    return start_time, exist, df

start_time, exist, df = get_start_time()

for num in range(config.num_of_download):
    print('downloading data from {}'.format(start_time))
    
    # download raw data
    data = client.Trade.Trade_getBucketed(symbol=config.symbol, reverse=False, binSize="1d", count=720, startTime = start_time).result()
    
    # increment start time
    start_time = start_time + timedelta(hours=12)
    
    # append data to dataframe
    if num == 0 and exist == 0:
        df = get_data_to_df(data)
    elif len(data[0]) != 0:
        df = pd.DataFrame.append(df, get_data_to_df(data))
    
    print("{}/{}, {} data collected".format(num, config.num_of_download, len(data[0])))
    
    if len(data[0]) < 720:
        print('job done, all data collected')
        break
    # sleep for 3 seconds to avoid frequent request ban
    time.sleep(3)
    
# save data to file
print('\nsaving file, this may take a while')
df.to_csv('{}/{}.csv'.format(config.save_path, config.symbol))
print('\nfinished')