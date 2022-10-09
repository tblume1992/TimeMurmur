# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from TimeMurmur.basis_functions.LinearBasisFunction import LinearBasisFunction

class PanelAxis:
    
    def __init__(self,
                 run_dict,
                 n_basis,
                 decay,
                 weighted,
                 seasonal_period):
        self.run_dict = run_dict
        self.seasonal_period = seasonal_period
        self.n_basis = n_basis
        self.decay = decay
        self.weighted = weighted
    
    def get_piecewise(self, y, ts_id):
        if self.n_basis >= len(y):
            n_basis = len(y) - 1
        else:
            n_basis = self.n_basis
        lbf = LinearBasisFunction(n_changepoints=n_basis,
                                  decay=self.decay,
                                  weighted=self.weighted)
        basis = lbf.get_basis(y)
        self.run_dict['local'][ts_id]['function'] = lbf
        self.run_dict['local'][ts_id]['basis'] = basis
        return basis

    def build_axis(self, dataset):
        ts_id = dataset['Murmur ID'].iloc[0]
        if self.n_basis is not None and self.n_basis:
            linear_basis = self.get_piecewise(dataset['Murmur Target'],
                                              ts_id)
            size = np.shape(linear_basis)[1] - 1
            linear_basis = pd.DataFrame(linear_basis,
                                        index=dataset.index,
                                        columns=[f'basis_{i}' for i in range(size)]+ ['Trend'])
            dataset = pd.concat([dataset, linear_basis], 
                                axis=1)
        return dataset
    
    def build_future_axis(self, refined_df, forecast_horizon, ts_id):
        id_dict = self.run_dict['local'][ts_id]
        if self.n_basis is not None and self.n_basis:
            X = id_dict['function'].get_future_basis(id_dict['basis'],
                                                      forecast_horizon)
            size = np.shape(X)[1] - 1
            X = pd.DataFrame(X,
                             index=refined_df.index,
                             columns=[f'basis_{i}' for i in range(size)] + ['Trend'])
            X['Murmur ID'] = ts_id
            X[self.run_dict['global']['Date Column']] = refined_df[self.run_dict['global']['Date Column']].values
            return X
        else:
            return None
