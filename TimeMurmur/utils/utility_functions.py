# -*- coding: utf-8 -*-
import numpy as np
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

def future_index(dataset, run_dict, freq, forecast_horizon):
    ts_id = dataset['Murmur ID'].iloc[0]
    date_column = run_dict['global']['Date Column']
    last_date = run_dict['global']['last date'][ts_id]
    future_index = pd.date_range(last_date,
                                 periods=forecast_horizon + 1,
                                 freq=freq)[1:]
    future_df = pd.DataFrame(future_index, columns=[date_column])
    future_df['Murmur ID'] = ts_id
    return future_df

def get_data():
    return pd.read_csv(r'./TimeMurmur/utils/murmur_example.csv')

# Error metrics from: https://github.com/kdgutier/esrnn_torch/blob/master/ESRNN/utils_evaluation.py
def mse(y, y_hat):
  """
  Calculates Mean Squared Error.
  
  Parameters
  ----------  
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------    
  mse: float
    mean squared error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mse = np.mean(np.square(y - y_hat))
  return mse

def mape(y, y_hat):
  """
  Calculates Mean Absolute Percentage Error.
  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------
  mape: float
    mean absolute percentage error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mape = np.mean(np.abs(y - y_hat) / (0.0001 + np.abs(y)))
  mape = min(2, mape)
  if mape == np.inf:
      mape = 2.0
  return mape

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (0.0001 + (np.abs(A) + np.abs(F))))

def mase(y, y_hat, y_train, seasonality):
  """
  Calculates Mean Absolute Scaled Error.
  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  y_train: numpy array
    actual train values for Naive1 predictions
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  
  Returns
  -------
  mase: float
    mean absolute scaled error
  """
  y_hat_naive = []
  for i in range(seasonality, len(y_train)):
      y_hat_naive.append(y_train[(i - seasonality)])

  masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
  mase = np.mean(abs(y - y_hat)) / masep
  return mase

