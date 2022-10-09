# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import numpy as np
from TimeMurmur.basis_functions.FourierBasisFunction import FourierBasisFunction


class TimeAxis:
    def __init__(self,
                 dates,
                 run_dict,
                 seasonal_period,
                 fourier_order, 
                 freq,
                 date_column,
                 seasonal_weights
                 ):
        self.dates = dates.drop_duplicates().sort_values().reset_index(drop=True)
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
        self.seasonal_period = seasonal_period
        self.fourier_order = fourier_order
        self.run_dict = run_dict
        self.freq = freq
        self.date_column = date_column
        self.seasonal_weights = seasonal_weights
    
    def get_fourier(self, length, fourier_order, seasonal_period):
        self.fbf = FourierBasisFunction(seasonal_weights=self.seasonal_weights)
        basis = self.fbf.get_basis(length, 
                              seasonal_period,
                              fourier_order)
        self.run_dict['global'][f'{seasonal_period}_basis_function'] = self.fbf
        return basis
    
    def get_future_fourier(self, forecast_horizon, seasonal_period, fourier_order): 
        length = len(self.run_dict['global']['Dataset Dates'])
        future_basis = self.fbf.get_future_basis(length, 
                                                 forecast_horizon, 
                                                 seasonal_period,
                                                 fourier_order)
        return future_basis
    
    @staticmethod
    def check_exogenous(exogenous):
        assert isinstance(exogenous, pd.DataFrame), 'Time Exogenous MUST be a DataFrame with a Datetime index'
        assert isinstance(exogenous.index, pd.DatetimeIndex), 'Time Exogenous MUST be a DataFrame with a Datetime index'
        
    def fill_gaps(self, dates):
        dates = pd.date_range(start=dates.iloc[0],
                              end=dates.iloc[-1],
                              freq=self.freq)
        return dates
    
    def build_future_index(self,
                           forecast_horizon):
        
        future_dates = pd.date_range(start=self.dates[-1],
                                     freq=self.freq,
                                     periods=forecast_horizon + 1)
        return future_dates[1:]    
    
    def build_axis(self, time_exogenous):
        self.dates = self.fill_gaps(self.dates)
        self.run_dict['global']['Dataset Dates'] = self.dates
        if time_exogenous is not None:
            TimeAxis.check_exogenous(time_exogenous)
        if self.seasonal_period is None and time_exogenous is None:
            self.run_dict['global']['seasonal_basis'] = None
            return None   
        if self.seasonal_period is None and time_exogenous is not None:
            self.run_dict['global']['seasonal_basis'] = None
            return time_exogenous
        axis = None
        for seas in self.seasonal_period:
            seas_axis = self.get_fourier(len(self.run_dict['global']['Dataset Dates']), 
                                    self.fourier_order,
                                    seas)
            seas_axis = pd.DataFrame(seas_axis,
                                columns=[f'{seas}_fourier_{i+1}' for i in range(2 * self.fourier_order)],
                                )
            if axis is None:
                axis = seas_axis
            else:
                axis = pd.concat([axis, seas_axis], axis=1)
        axis[self.date_column] = self.dates
        self.run_dict['global']['seasonal_basis'] = axis
        if time_exogenous is not None:
            axis = axis.merge(time_exogenous, 
                              on=self.date_column,
                              how='left')
        return axis
    
    def build_future_axis(self, 
                          future_dates,
                          forecast_horizon,
                          future_exogenous):
        if future_exogenous is not None:
            TimeAxis.check_exogenous(future_exogenous)
        seasonal_basis = self.run_dict['global']['seasonal_basis']
        if seasonal_basis is None and future_exogenous is None:
            return None
        if seasonal_basis is None and future_exogenous is not None:
            return future_exogenous
        future_axis = None
        for seas in self.seasonal_period:
            self.fbf = self.run_dict['global'][f'{seas}_basis_function']
            future_seas = self.get_future_fourier(forecast_horizon,
                                                  seas,
                                                  self.fourier_order)
            future_seas = pd.DataFrame(future_seas,
                                columns=[f'{seas}_fourier_{i+1}' for i in range(2 * self.fourier_order)],
                                index=future_dates)
            if future_axis is None:
                future_axis = future_seas
            else:
                future_axis = pd.concat([future_axis, future_seas], axis=1)
        if future_exogenous is not None:
            future_axis = future_axis.merge(future_exogenous, 
                                            left_index=True, 
                                            right_index=True,
                                            how='left')
        return future_axis     
            