# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder  
from TimeMurmur.utils.utility_functions import infer_freq, handle_future_index
from TimeMurmur.builder.PreProcess import PreProcess
from TimeMurmur.builder.BuildTimeAxis import TimeAxis
from TimeMurmur.builder.BuildPanelAxis import PanelAxis
from TimeMurmur.builder.BuildArAxis import ArAxis
from TimeMurmur.builder.BuildIdAxis import IdAxis
from TimeMurmur.builder.ExtractFeatures import ExtractFeatures
tqdm.pandas()


class Builder:
    def __init__(self,
                 target_column,
                 id_column,
                 date_column,
                 scale,
                 ar,
                 ma,
                 decay,
                 n_basis,
                 seasonal_period,
                 difference,
                 freq,
                 weighted,
                 categorical_columns,
                 linear_trend,
                 fourier_order,
                 seasonal_weights,
                 test_size,
                 scale_type,
                 basis_difference,
                 linear_test_window,
                 seasonal_dummy,
                 floor_bind,
                 floor,
                 outlier_cap,
                 ts_features):
        if isinstance(ar, int):
            raise ValueError('AR Lag must be passed as a list!')
        if isinstance(ma, int):
            raise ValueError('MA Lag must be passed as a list!')
        self.ar = ar
        self.ma = ma
        self.target_column = target_column
        self.id_column = id_column 
        self.date_column = date_column
        self.freq = freq
        self.scale = scale 
        self.fourier_order = fourier_order
        self.weighted = weighted
        self.linear_test_window = linear_test_window
        self.seasonal_weights = seasonal_weights
        self.difference = difference 
        self.n_basis = n_basis
        self.seasonal_dummy = seasonal_dummy
        self.decay = decay
        self.basis_difference = basis_difference
        self.scale_type = scale_type
        if isinstance(seasonal_period, int) or isinstance(seasonal_period, float):
            seasonal_period = [seasonal_period]
        self.seasonal_period = seasonal_period
        self.categorical_columns = categorical_columns 
        self.test_size = test_size
        self.linear_trend = linear_trend
        self.floor_bind = floor_bind
        self.floor = floor
        self.outlier_cap = outlier_cap
        self.ts_features = ts_features
        self.run_dict = {}
        self.run_dict['local'] = {}
        self.run_dict['global'] = {}
        self.run_dict['global']['categorical_encoder'] = {}
        self.run_dict['global']['IDs with Trend'] = []
        self.run_dict['global']['main_seasonal_period'] = self.seasonal_period[0]
        self.run_dict['global']['id_axis'] = None
        self.run_dict['global']['Date Column'] = date_column
        self.run_dict['global']['ID Column'] = id_column
        self.run_dict['global']['Target Column'] = target_column
        self.run_dict['global']['ts_features'] = None
        if ar is None:
            self.run_dict['global']['AR Datasets'] = None
        else:
            self.run_dict['global']['AR Datasets'] = {}
        if ma is None:
            self.run_dict['global']['MA Datasets'] = None
            self.run_dict['global']['Future MA Datasets'] = None
        else:
            self.run_dict['global']['MA Datasets'] = {}
            self.run_dict['global']['Future MA Datasets'] = {}

    def build_single(self, single_dataset):
        ts_id = single_dataset['Murmur ID'].iloc[0]
        self.run_dict['local'][ts_id] = {}
        single_dataset = PreProcess(target_column=self.target_column,
                                    scale=self.scale,
                                    difference=self.difference,
                                    test_size=self.test_size,
                                    run_dict=self.run_dict,
                                    seasonal_period=self.seasonal_period,
                                    linear_trend=self.linear_trend,
                                    scale_type=self.scale_type,
                                    linear_test_window=self.linear_test_window,
                                    floor_bind=self.floor_bind,
                                    floor=self.floor,
                                    outlier_cap=self.outlier_cap).process(single_dataset)
        self.single = single_dataset
        single_dataset = self.build_panel_axis(single_dataset)
        return single_dataset
    
    def build_single_prediction(self, single_dataset):
        single_dataset = single_dataset.reset_index()
        ts_id = single_dataset['Murmur ID'].iloc[0]
        single_dataset = self.build_future_panel_axis(single_dataset, ts_id)
        return single_dataset
    
    def create_murmur_id(self, dataset):
        try:
            dataset[self.id_column] = dataset[self.id_column].astype(int)
            dataset['Murmur ID'] = dataset[self.id_column]
            self.run_dict['global']['ID Mapping'] = dataset[[self.id_column, 'Murmur ID']].drop_duplicates()
        except:
            le = LabelEncoder()
            dataset['Murmur ID'] = le.fit_transform(dataset[self.id_column])
        self.run_dict['global']['ID Mapping'] = dataset[[self.id_column, 'Murmur ID']].drop_duplicates()
        return dataset
    
    def handle_categorical(self, dataset, categorical_columns):
        for column in categorical_columns:
            try:
                dataset[column].astype(int)
                self.run_dict['global']['categorical_encoder'][column] = None
            except Exception:
                try:
                    le = LabelEncoder()
                    le.fit(dataset[column])
                    dataset[column] =  le.transform(dataset[column])
                    self.run_dict['global']['categorical_encoder'][column] = le
                except:
                    dataset[column] = dataset[column].fillna('nan')
                    le = LabelEncoder()
                    le.fit(dataset[column])
                    dataset[column] =  le.transform(dataset[column])
                    self.run_dict['global']['categorical_encoder'][column] = le                   
            dataset[column] = dataset[column].astype('category')
        return dataset
    
    def check_input(self, dataset):
        assert len(dataset[[self.id_column, self.date_column]])
        required_columns = [self.id_column,
                           self.target_column,
                           self.date_column]
        if self.categorical_columns is not None:
            required_columns += self.categorical_columns
        columns = dataset.columns
        for column in required_columns:
            assert column in columns, f'Missing {column} from dataset'
        return required_columns
    
    def build_time_axis(self, dataset, time_exogenous=None):
        dates = dataset[self.date_column].drop_duplicates().sort_values()
        if self.freq == 'auto':
            dates
            self.freq = infer_freq(dates)
        self.run_dict['global']['freq'] = self.freq
        time_builder = TimeAxis(dates=dataset[self.date_column], 
                                run_dict=self.run_dict,
                                seasonal_period=self.seasonal_period,
                                fourier_order=self.fourier_order,
                                date_column=self.date_column,
                                seasonal_weights=self.seasonal_weights,
                                freq=self.freq,
                                seasonal_dummy=self.seasonal_dummy
                                )
        self.run_dict['global']['Dates'] = dates
        time_axis = time_builder.build_axis(time_exogenous=time_exogenous)
        return time_axis
    
    def build_future_time_axis(self, forecast_horizon, dataset, future_time_exogenous=None):
        dates = dataset[self.date_column].drop_duplicates().sort_values()
        time_builder = TimeAxis(dates=dataset[self.date_column], 
                                run_dict=self.run_dict,
                                seasonal_period=self.seasonal_period,
                                fourier_order=self.fourier_order,
                                date_column=self.date_column,
                                seasonal_weights=self.seasonal_weights,
                                freq=self.freq,
                                seasonal_dummy=self.seasonal_dummy
                                )
        self.run_dict['global']['Dates'] = dates
        time_axis = time_builder.build_future_axis(dates,
                                                   forecast_horizon=forecast_horizon,
                                                   future_exogenous=future_time_exogenous)
        return time_axis
    
    def build_id_axis(self, dataset, id_exogenous=None, feature_columns=None):
        if id_exogenous is not None:
            if list(set(id_exogenous.columns).intersection(set(self.categorical_columns))):
                id_exogenous = self.handle_categorical(id_exogenous,
                                                       self.categorical_columns)
        id_builder = IdAxis(id_column=self.id_column,
                            feature_columns=feature_columns)
        id_axis = id_builder.build_axis(dataset,
                                        id_exogenous=id_exogenous)
        id_axis = id_axis.merge(self.run_dict['global']['ID Mapping'],
                                on=self.id_column)
        self.run_dict['global']['id_axis'] = id_axis
        return id_axis
    
    def build_ar_axis(self, dataset):
        ar_dataset = dataset[['Murmur ID', self.date_column, 'Murmur Target']]
        ar_builder = ArAxis(self.run_dict,
                            self.freq)
        ar_builder.build_axis(ar_dataset, self.ar, self.ma)
        return
    
    def build_future_ar_axis(self, predicted_dataset):
        ar_builder = ArAxis(self.run_dict,
                            self.freq)
        ar_builder.build_future_axis(predicted_dataset, self.ar)
        return
        
    def build_panel_axis(self, dataset):
        panel_builder = PanelAxis(run_dict=self.run_dict,
                                  n_basis=self.n_basis,
                                  decay=self.decay,
                                  weighted=self.weighted,
                                  seasonal_period=self.seasonal_period,
                                  basis_difference=self.basis_difference)
        panel_axis = panel_builder.build_axis(dataset)
        return panel_axis

    def build_future_panel_axis(self, dataset, ts_id):
        panel_builder = PanelAxis(run_dict=self.run_dict,
                                  # ar=self.ar,
                                  n_basis=self.n_basis,
                                  decay=self.decay,
                                  weighted=self.weighted,
                                  seasonal_period=self.seasonal_period,
                                  basis_difference=self.basis_difference)
        panel_axis = panel_builder.build_future_axis(dataset, 
                                                     self.forecast_horizon,
                                                     ts_id)
        return panel_axis
    
    def preprocess(self, dataset):
        required_columns = self.check_input(dataset)
        dataset = dataset.copy(deep=True)[required_columns]
        dataset = self.create_murmur_id(dataset)
        dataset = dataset.sort_values(self.date_column)
        if self.categorical_columns is not None:
            dataset = self.handle_categorical(dataset, 
                                              self.categorical_columns)
        dataset = dataset.groupby(by='Murmur ID').progress_apply(self.build_single)
        return dataset
    
    def build_dataset(self, 
                      dataset,
                      time_axis=None,
                      id_axis=None):
        if 'Murmur ID' in dataset.columns:
            processed_dataset = dataset
        else:
            print('Preprocessing Data: Scaling, building basis functions, capping outliers etc.')
            processed_dataset = self.preprocess(dataset)
        if id_axis is not None:
            print('Building ID Features')
            try:
                drop_columns = [i for i in list(id_axis.columns) if i not in ['Murmur ID',
                                                                              self.id_column]]
                processed_dataset = processed_dataset.drop(drop_columns,
                                                           axis=1)
            except Exception:
                pass
            processed_dataset = processed_dataset.merge(id_axis,
                                                        on=[self.id_column,
                                                            'Murmur ID'],
                                                        how='left')
        if self.ts_features:
            print('Building Time Series Features')
            ExtractFeatures(self.run_dict).build_axis(processed_dataset)
            processed_dataset = processed_dataset.merge(self.run_dict['global']['ts_features'],
                                    on='Murmur ID',
                                    how='left')
        if time_axis is not None:
            print('Building Time Features')
            processed_dataset = processed_dataset.merge(time_axis,
                                                        on=self.date_column,
                                                        how='left')
        if self.ar is not None:
            print('Building AR Lags')
            self.build_ar_axis(processed_dataset)
            for _, ar_dataset in tqdm(self.run_dict['global']['AR Datasets'].items()):
                processed_dataset = processed_dataset.merge(ar_dataset,
                                                            on=['Murmur ID', self.date_column],
                                                            how='left')
        if self.ma is not None:
            print('Building MA Lags')
            self.build_ar_axis(processed_dataset)
            for _, ma_dataset in tqdm(self.run_dict['global']['MA Datasets'].items()):
                processed_dataset = processed_dataset.merge(ma_dataset,
                                                            on=['Murmur ID', self.date_column],
                                                            how='left')
        return processed_dataset
    
    def build_future_dataset(self, 
                             forecast_horizon,                              
                             time_exogenous=None, 
                             panel_exogenous=None): 
        self.forecast_horizon = forecast_horizon
        date_column = self.run_dict['global']['Date Column']
        id_column = 'Murmur ID'
        merge_on = [id_column, date_column]
        id_df = pd.DataFrame(self.run_dict['local'].keys(),
                             columns=[id_column])
        forecast_dates = handle_future_index(self.run_dict['global']['Dates'],
                                             freq=self.run_dict['global']['freq'],
                                             forecast_horizon=forecast_horizon)
        date_df = pd.DataFrame(forecast_dates,
                               columns=[date_column])
        self.run_dict['global']['Forecast Dates'] = forecast_dates
        pred_X = pd.merge(id_df, date_df, how='cross')  
        time_axis = self.build_future_time_axis(forecast_horizon,
                                                pred_X,
                                                time_exogenous)
        if self.n_basis and self.n_basis is not None:
            pred_X = pred_X.groupby('Murmur ID').apply(self.build_single_prediction)
            pred_X = pred_X.reset_index(drop=True)
        if self.run_dict['global']['ts_features'] is not None:
            print('Building Time Series Features')
            pred_X = pred_X.merge(self.run_dict['global']['ts_features'],
                                  on='Murmur ID',
                                  how='left')
        if time_axis is not None:
            pred_X = pred_X.merge(time_axis, 
                                  left_on=self.date_column,
                                  right_index=True,
                                  how='left')
        if self.run_dict['global']['id_axis'] is not None:
            pred_X = pred_X.merge(self.run_dict['global']['id_axis'],
                                    on='Murmur ID',
                                    how='left')
        # if self.run_dict['global']['ts_features'] is not None:
        #     pred_X = pred_X.merge(self.run_dict['global']['ts_features'],
        #                             on='Murmur ID',
        #                             how='left')
        if self.run_dict['global']['Future MA Datasets'] is not None:
            for ma in self.ma:
                ma_df = self.run_dict['global']['Future MA Datasets'][f'ma_{ma}']
                pred_X = pred_X.merge(ma_df,
                                      on='Murmur ID',
                                      how='left')
        # if time_exogenous is not None:
        #     pred_X = pred_X.merge(time_exogenous, 
        #                           on=self.date_column,
        #                           how='left')
        if panel_exogenous is not None:
            pred_X = pred_X.merge(panel_exogenous,
                                    on=merge_on,
                                    how='left')
        if self.categorical_columns is not None:
            pred_X = self.handle_categorical(pred_X, 
                                             self.categorical_columns)

        return pred_X
