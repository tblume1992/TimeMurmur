# -*- coding: utf-8 -*-
import pandas as pd


class ArAxis:
    def __init__(self, run_dict, freq):
        self.run_dict = run_dict
        self.freq = freq
    
    def build_axis(self, 
                 dataset, 
                 ar_order):
        id_column = 'Murmur ID'
        date_column = self.run_dict['global']['Date Column']
        target_column = 'Murmur Target'
        dataset.index = dataset[date_column]
        for ar in ar_order:
            ar_df = dataset[[id_column,
                             target_column]]
            ar_df = ar_df.rename({target_column: f'ar_{ar}'}, axis=1) 
            self.run_dict['global']['AR Datasets'][f'ar_{ar}'] = ar_df.shift(periods=ar, 
                                                                 freq=self.freq).reset_index()
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