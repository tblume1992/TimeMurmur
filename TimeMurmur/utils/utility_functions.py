# -*- coding: utf-8 -*-

import pandas as pd


def infer_freq(dates):
    try:
        dates.index = dates
        freq = pd.infer_freq(dates.index)
        if freq is None:
            raise ValueError('Frequency cannot be determined, please pass freq to forecast method') 
    except Exception:
        raise ValueError('Error inferring frequency from date column, ensure it is datetime with df[date_column] = pd.to_datetime(df[date_column])')
    return freq

def handle_future_index(dates, freq, forecast_horizon):
    last_date = dates.iloc[-1]
    future_index = pd.date_range(last_date,
                                 periods=forecast_horizon + 1,
                                 freq=freq)[1:]
    return future_index

def get_data():
    return pd.read_csv(r'TimeMurmur/utils/murmur_example.csv')