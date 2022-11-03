# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   PowerTransformer, MaxAbsScaler, QuantileTransformer)


class PreProcess:
    def __init__(self,
                 target_column,
                 scale,
                 difference,
                 test_size,
                 linear_trend,
                 seasonal_period,
                 run_dict,
                 scale_type,
                 linear_test_window,
                 floor_bind,
                 floor,
                 outlier_cap):
        self.target_column = target_column
        self.scale = scale
        self.difference = difference
        self.test_size = test_size 
        self.seasonal_period = seasonal_period
        self.run_dict = run_dict
        self.linear_trend = linear_trend
        self.scale_type = scale_type
        self.linear_test_window = linear_test_window
        self.floor_bind = floor_bind
        self.floor = floor
        self.outlier_cap = outlier_cap
        
    def scale_input(self, y, ts_id):
        if self.scale_type == 'standard':
            scaler = StandardScaler()
        elif self.scale_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.scale_type == 'maxabs':
            scaler = MaxAbsScaler()
        elif self.scale_type == 'robust':
            scaler = RobustScaler()
        elif self.scale_type == 'quantile':
            scaler = QuantileTransformer()
        elif self.scale_type == 'boxcox':
            scaler = PowerTransformer(method='box-cox')
        #unstable
        # elif self.scale_type == 'power':
        #     scaler = PowerTransformer()
        else:
            raise ValueError(f'{self.scale_type} not recognized! Options are: standard or minmax')
        scaler.fit(np.asarray(y).reshape(-1, 1))
        self.run_dict['local'][ts_id]['scaler'] = scaler
        scaled_y = y.values.copy()
        scaled_y = scaler.transform(scaled_y.reshape(-1, 1))
        return scaled_y
        
    def create_test_set(self, dataset, test_size):
        n = len(dataset)
        if test_size is None or not test_size:
            return ['Train'] * n
        else:
            if test_size < 1:
                train_size = int(n * (1 - test_size))
                test_size = n - train_size
            else:
                train_size = n - test_size
                if train_size < 0:
                    train_size = n
                    test_size = 0
            return ['Train'] * train_size + ['Test'] * test_size

    def linear_test(self, y):
        y = y.values
        xi = np.arange(1, len(y) + 1)
        # xi = xi**2
        slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
        trend_line = slope*xi*r_value + intercept
        if self.seasonal_period is not None:
            required_len = 1.5 * max(self.seasonal_period)
        else:
            required_len = 6
        if self.linear_trend and len(y) > required_len:
            if self.linear_test_window is not None:
                n_bins = self.linear_test_window
            else:
                n_bins = (1 + len(y)**(1/3) * 2)
            splitted_array = np.array_split(y.reshape(-1,), int(n_bins))
            mean_splits = np.array([np.mean(i) for i in splitted_array])
            asc_array = np.sort(mean_splits)
            desc_array = np.flip(asc_array)
            if all(asc_array == mean_splits):
                growth = True
            elif all(desc_array == mean_splits):
                growth = True
            else:
                growth = False
            if (r_value > .9 and growth):
                linear = True
            else:
                linear = False
        else:
            linear = False
        slope = slope * r_value
        return trend_line, linear, slope, intercept, r_value

    def cap_outliers(self, series, outlier_cap):
        mean = series.mean()
        std = np.std(series)
        series = series.clip(lower=mean - outlier_cap * std,
                             upper=mean + outlier_cap * std)
        return series

    def process(self, dataset):
        ts_id = dataset['Murmur ID'].iloc[0]
        self.run_dict['local'][ts_id]['trend'] = {}
        # if self.difference:
        #     dataset[self.target_column] = dataset[self.target_column].diff(periods=1)
        if self.scale:
            if self.outlier_cap is not None:
                dataset[self.target_column] = self.cap_outliers(dataset[self.target_column],
                                                                self.outlier_cap)
            dataset['Murmur Target'] = self.scale_input(dataset[self.target_column],
                                                        ts_id)
            if self.floor_bind:
                idxs = dataset[dataset[self.target_column] == self.floor].index
                dataset.loc[idxs]['Murmur Target'] = dataset['Murmur Target'].min()
        else:
            dataset['Murmur Target'] = dataset[self.target_column]
            self.run_dict['local'][ts_id]['scaler'] = None
        if self.linear_trend :
            trend_line, linear, slope, intercept, penalty = self.linear_test(dataset['Murmur Target'])
            if linear or self.linear_trend==True:
                self.run_dict['global']['IDs with Trend'].append(ts_id)
                dataset['Murmur Target'] = np.subtract(dataset['Murmur Target'].values, 
                                                       trend_line)
                self.run_dict['local'][ts_id]['trend']['trend_line'] = trend_line
                self.run_dict['local'][ts_id]['trend']['slope'] = slope
                self.run_dict['local'][ts_id]['trend']['intercept'] = intercept
                self.run_dict['local'][ts_id]['trend']['penalty'] = penalty
            else:
                self.run_dict['local'][ts_id]['trend']['trend_line'] = None
        else:
            self.run_dict['local'][ts_id]['trend']['trend_line'] = None
        dataset['Murmur Data Split'] = self.create_test_set(dataset, self.test_size)
        return dataset
