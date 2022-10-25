# -*- coding: utf-8 -*-
import pandas as pd


class ArAxis:
    def __init__(self, run_dict, freq):
        self.run_dict = run_dict
        self.freq = freq
    
    def build_axis(self, 
                 dataset, 
                 ar_order,
                 ma_order):
        id_column = 'Murmur ID'
        date_column = self.run_dict['global']['Date Column']
        target_column = 'Murmur Target'
        dataset.index = dataset[date_column]
        if ar_order is not None:
            for ar in ar_order:
                ar_df = dataset[[id_column,
                                 target_column]]
                ar_df = ar_df.rename({target_column: f'ar_{ar}'}, axis=1) 
                self.run_dict['global']['AR Datasets'][f'ar_{ar}'] = ar_df.shift(periods=ar, 
                                                                                 freq=self.freq).reset_index()
        if ma_order is not None:
            for ma in ma_order:
                ma_df = dataset[[id_column,
                                 date_column,
                                 target_column]]
                # ma_df = ma_df.rename({target_column: f'ma_{ma}'}, axis=1)
                max_date = ma_df[date_column].max()
                # lag_date = max_date - pd.Timedelta(ma, self.freq)
                # future_df = ma_df[ma_df[date_column] >= max_date]
                # future_df = future_df.pivot(index='Murmur ID',
                #                             columns='Datetime',
                #                             values='Murmur Target')
                # future_df = future_df.reset_index(drop=True)
                ma_df[f'ma_{ma}'] = ma_df['Murmur Target'].rolling(ma).mean()
                ma_df = ma_df.reset_index(drop=True)
                future_df = ma_df[ma_df[date_column] == max_date]
                future_df = future_df.drop(['Murmur Target', date_column], axis=1)
                self.run_dict['global']['MA Datasets'][f'ma_{ma}'] = ma_df.drop('Murmur Target',
                                                                                axis=1)
                self.run_dict['global']['Future MA Datasets'][f'ma_{ma}'] = future_df
        return
    
    def build_future_axis(self, 
                 dataset, 
                 ar_order):
        id_column = 'Murmur ID'
        date_column = self.run_dict['global']['Date Column']
        target_column = 'Predictions'
        dataset.index = dataset[date_column]
        for ar in ar_order:
            ar_df = dataset[[id_column,
                             target_column]]
            og_dataset = self.run_dict['global']['AR Datasets'][f'ar_{ar}'] 
            shifted_dataset = ar_df.shift(periods=ar, 
                                          freq=self.freq)
            shifted_dataset = shifted_dataset.rename({target_column: f'ar_{ar}'}, 
                                                     axis=1).reset_index() 
            ar_df.rename({target_column: f'ar_{ar}'}, axis=1) 
            self.run_dict['global']['AR Datasets'][f'ar_{ar}'] = pd.concat([og_dataset,
                                                                            shifted_dataset])
            del shifted_dataset
            del ar_df
        return