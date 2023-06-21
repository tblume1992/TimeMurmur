# -*- coding: utf-8 -*-
import numpy as np
from TimeMurmur.builder.Transformer import MurmurScaler


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
        scaler = MurmurScaler(self.scale_type,
                              self.scale,
                              self.difference,
                              self.linear_trend,
                              self.linear_test_window,
                              self.seasonal_period,
                              self.run_dict,
                              ts_id)
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

    def cap_outliers(self, series, outlier_cap):
        """
        

        Parameters
        ----------
        series : TYPE
            DESCRIPTION.
        outlier_cap : TYPE
            DESCRIPTION.

        Returns
        -------
        series : TYPE
            DESCRIPTION.

        """
        mean = series.mean()
        std = np.std(series)
        series = series.clip(lower=mean - outlier_cap * std,
                             upper=mean + outlier_cap * std)
        return series

    def process(self, dataset):
        """
        

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.

        Returns
        -------
        dataset : TYPE
            DESCRIPTION.

        """
        ts_id = dataset['Murmur ID'].iloc[0]
        self.run_dict['local'][ts_id]['trend'] = {}
        dataset['Murmur Target'] = dataset[self.target_column]
        if self.scale or self.difference or self.linear_trend:
            # print('scaling or differencing')
            if self.outlier_cap is not None:
                dataset['Murmur Target'] = self.cap_outliers(dataset['Murmur Target'],
                                                                self.outlier_cap)
            dataset['Murmur Target'] = self.scale_input(dataset['Murmur Target'],
                                                        ts_id)
            if self.floor_bind:
                idxs = dataset[dataset['Murmur Target'] == self.floor].index
                dataset.loc[idxs]['Murmur Target'] = dataset['Murmur Target'].min()
        else:
            self.run_dict['local'][ts_id]['scaler'] = None
        dataset['Murmur Data Split'] = self.create_test_set(dataset, self.test_size)
        # dataset = dataset.dropna(subset=['Murmur Target'])
        return dataset
