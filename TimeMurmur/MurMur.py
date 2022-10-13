# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from TimeMurmur.Optimizer import Optimize
from TimeMurmur.builder.Builder import Builder
from TimeMurmur.Model import Model
from TimeMurmur.utils.utility_functions import infer_freq
sns.set_style('darkgrid')


class Murmur:
    def __init__(self,
                 floor=None):
        self.floor = floor
        self.run_dict = None
        self.builder = None
    
    def fit(self,
            df,
            target_column,
            id_column,
            date_column,
            freq,
            alpha=None,
            time_exogenous=None,
            id_exogenous=None,
            id_feature_columns=None,
            time_feature_columns=None,
            scale=True,
            difference=False,
            categorical_columns=None,
            decay=.99,
            ar=None,
            fourier_order=10,
            seasonal_weights=None,
            weighted=True,
            n_basis=10,
            seasonal_period=None,
            test_size=None,
            linear_trend='auto',
            objective='regression',
            metric='rmse',
            learning_rate=.1,
            min_child_samples=5,
            num_leaves=50,
            num_iterations=50,
            return_proba=False,
            boosting_params=None,
            early_stopping_rounds=10,
            is_unbalance=True,
            scale_pos_weights=None,
            labels=None,
            floor_bind=False
            ):
        self.scale = scale
        self.id_column = id_column
        if freq == 'auto':
            dates = df[date_column].drop_duplicates().sort_values()
            freq = infer_freq(dates)
        self.builder = Builder(target_column=target_column,
                          id_column=id_column,
                          date_column=date_column,
                          scale=scale,
                          difference=difference,
                          categorical_columns=categorical_columns,
                          decay=decay,
                          ar=ar,
                          fourier_order=fourier_order,
                          freq=freq,
                          seasonal_weights=seasonal_weights,
                          weighted=weighted,
                          n_basis=n_basis,
                          seasonal_period=seasonal_period,
                          test_size=test_size,
                          linear_trend=linear_trend)
        process_dataset = self.builder.preprocess(df)
        if id_feature_columns is not None or id_exogenous is not None:
            id_dataset = self.builder.build_id_axis(process_dataset,
                                               id_exogenous=id_exogenous,
                                               feature_columns=id_feature_columns)
            if id_feature_columns is not None:
                id_feature_columns = [i for i in id_feature_columns if i != id_column]
                process_dataset = process_dataset.drop(id_feature_columns, axis=1)

        else:
            id_dataset = None
        time_dataset = self.builder.build_time_axis(df,
                                               time_exogenous=time_exogenous)
        dataset = self.builder.build_dataset(process_dataset, 
                                        id_axis=id_dataset,
                                        time_axis=time_dataset)
        drop_columns = [id_column, date_column, target_column]
        self.run_dict = self.builder.run_dict
        
        model = Model(objective=objective,
                      metric=metric,
                      learning_rate=learning_rate,
                      min_child_samples=min_child_samples,
                      num_leaves=num_leaves,
                      num_iterations=num_iterations,
                      return_proba=return_proba,
                      boosting_params=boosting_params,
                      scale_pos_weights=scale_pos_weights,
                      is_unbalance=is_unbalance,
                      alpha=alpha)
        self.model_obj = model.build_model()
        self.run_dict['global']['model'] = model.boosting_params
        if categorical_columns is None:
            cat_features = ['Murmur ID']
        else:
            cat_features = list(set(categorical_columns + ['Murmur ID']))
        dataset = dataset.sort_values(by=['Murmur ID', date_column])
        train_X = dataset.drop(drop_columns+['Murmur Target', 'Murmur Data Split'], axis=1)
        eval_set = []
        train_y = dataset['Murmur Target']
        if labels is not None:
            train_y = labels
        if not eval_set:
            early_stopping_rounds = None
        self.columns = train_X.columns
        self.train_X = dataset
        self.train_y = train_y
        self.dataset = dataset
        self.model_obj.fit(train_X,
                           train_y, 
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds, 
                           categorical_feature=cat_features)
        fitted = self.model_obj.predict(train_X)
        fitted_df = dataset[['Murmur ID', id_column, date_column, target_column]]
        fitted_df['Predictions'] = fitted
        if self.scale:
            fitted_df = self.unscale(fitted_df)
        fitted_df = self.retrend_fitted(fitted_df)
        if self.floor is not None:
            fitted_df['Predictions'] = fitted_df['Predictions'].clip(lower=self.floor)
        return fitted_df
    
    def ar_predict(self, forecast_horizon, pred_X):
        date_column = self.run_dict['global']['Date Column']
        id_column = 'Murmur ID'
        final_predicted = []
        print('Running Recursive Predictions')
        forecast_dates = self.run_dict['global']['Forecast Dates'].sort_values()
        for date in tqdm(forecast_dates):
            refined_pred_X = pred_X[pred_X[date_column] == date]

            for _, dataset in self.run_dict['global']['AR Datasets'].items(): 
                refined_pred_X = refined_pred_X.merge(dataset,
                                                      on=['Murmur ID',
                                                          date_column],
                                                      how='left')
            self.refined_pred_X = refined_pred_X
            model_X = refined_pred_X.drop([date_column], axis=1)
            predicted = self.model_obj.predict(model_X[list(self.columns)])
            predicted = pd.DataFrame(predicted, columns=['Predictions'])
            predicted[id_column] = refined_pred_X[id_column].values
            predicted[date_column] = refined_pred_X[date_column].values
            self.builder.build_future_ar_axis(predicted)
            final_predicted.append(predicted)
        return pd.concat(final_predicted)
    
    def predict(self, 
                forecast_horizon,
                time_exogenous=None,
                panel_exogenous=None,
                predict_proba=False
                ):
        date_column = self.run_dict['global']['Date Column']
        id_column = self.run_dict['global']['ID Column']
        drop_columns = [date_column]
        pred_X = self.builder.build_future_dataset(forecast_horizon,
                                                   time_exogenous=time_exogenous,
                                                   panel_exogenous=panel_exogenous)
        self.pred_X = pred_X
        if self.run_dict['global']['AR Datasets'] is not None:
            predicted_df = self.ar_predict(forecast_horizon,
                                        pred_X)
            predicted_df = predicted_df.merge(self.run_dict['global']['ID Mapping'],
                                              on='Murmur ID',
                                              how='left')
        else:
            if predict_proba:
                predicted = self.model_obj.predict_proba(pred_X[list(self.columns)])
                predicted = predicted[:, 0]
            else:
                predicted = self.model_obj.predict(pred_X[list(self.columns)])
            predicted_df = pred_X[['Murmur ID', date_column]]
            predicted_df = predicted_df.merge(self.run_dict['global']['ID Mapping'],
                                              on='Murmur ID',
                                              how='left')
            predicted_df['Predictions'] = predicted

        if self.scale:
            predicted_df = self.unscale(predicted_df)
        predicted_df = self.retrend_predicted(predicted_df)
        if self.floor is not None:
            predicted_df['Predictions'] = predicted_df['Predictions'].clip(lower=self.floor)
        return predicted_df

    def inverse_transform(self, df):
        scaler = self.run_dict['local'][df['Murmur ID'].iloc[0]]['scaler']
        df['Predictions'] = scaler.inverse_transform(df['Predictions'].values.reshape(-1,1))
        return df
    
    def unscale(self, forecast_df):
        return forecast_df.groupby('Murmur ID').apply(self.inverse_transform)


    def retrend_fitted(self, fitted):
        trend_ids = self.run_dict['global']['IDs with Trend']
        if trend_ids:
            for ts_id in tqdm(trend_ids):
                y = fitted[fitted['Murmur ID'] == int(ts_id)]['Predictions']
                trend = self.run_dict['local'][ts_id]['trend']['trend_line']
                fitted.loc[fitted['Murmur ID'] == int(ts_id),'Predictions'] = y + trend
        return fitted

    def retrend_predicted(self, predicted):
        trend_ids = self.run_dict['global']['IDs with Trend']
        if trend_ids:
            for ts_id in tqdm(trend_ids):
                y = predicted[predicted['Murmur ID'] == int(ts_id)]['Predictions'].values
                slope = self.run_dict['local'][ts_id]['trend']['slope']
                intercept = self.run_dict['local'][ts_id]['trend']['intercept']
                penalty = self.run_dict['local'][ts_id]['trend']['penalty']
                n = len(self.run_dict['local'][ts_id]['trend']['trend_line'])
                linear_trend = [i for i in range(0, len(y))]
                linear_trend = np.reshape(linear_trend, (len(linear_trend), 1))
                linear_trend += n + 1
                linear_trend = np.multiply(linear_trend, slope*penalty) + intercept
                retrended_pred = y + np.reshape(linear_trend, (-1,))
                predicted.loc[predicted['Murmur ID'] == int(ts_id),'Predictions'] = retrended_pred
        return predicted

    def Explain(self):
        explainer = shap.TreeExplainer(self.model_obj)
        shap_values = explainer.shap_values(self.train_X)
        return explainer, shap_values

    def Optimize(cls, y, seasonality, n_folds, test_size=None):
        optimizer = Optimize(y, Murmur, seasonality, n_folds, test_size)
        optimized = optimizer.fit()
        optimized['ar'] = list(range(1, int(optimized['ar']) + 1))
        optimized['n_estimators'] = int(optimized['n_estimators'])
        optimized['num_leaves'] = int(optimized['num_leaves'])
        optimized['n_basis'] = int(optimized['n_basis'])
        optimized['fourier_order'] = int(optimized['fourier_order'])
        return optimized

    def plot(self,
             fitted,
             ts_id=None,
             murmur_id=None,
             level=None,
             predicted=None,
             upper_fitted=None,
             upper_predicted=None,
             lower_fitted=None,
             lower_predicted=None):
        if (murmur_id is None and ts_id is None) and level is None:
            raise ValueError('Must pass a level or time series ID')
        date_column = self.run_dict['global']['Date Column']
        if murmur_id is not None:
            id_column = 'Murmur ID'
            ts_id = murmur_id
        else:
            id_column = self.run_dict['global']['ID Column']
        target_column = self.run_dict['global']['Target Column']
        if level == 'all':
            refined_df = fitted.groupby(date_column)[[target_column,'Predictions']].sum().reset_index()
        else:
            refined_df = fitted[fitted[id_column] == ts_id]
        plt.plot(refined_df[date_column],
                 refined_df['Predictions'],  
                 color='lightseagreen')
        if predicted is not None:
            if level is not None:
                predicted = predicted.groupby(date_column)[['Predictions']].sum().reset_index()
            else:
                predicted = predicted[predicted[id_column] == ts_id]
            plt.plot(predicted[date_column],
                     predicted['Predictions'],
                     color='lightseagreen',
                     linestyle='dashed')
        plt.plot(refined_df[date_column],
                  refined_df[target_column],
                  color='navy')
        if upper_fitted is not None and lower_fitted is not None:
            if level is not None:
                upper_df = upper_fitted.groupby(date_column)[[target_column,'Predictions']].sum().reset_index()
                lower_df = lower_fitted.groupby(date_column)[[target_column,'Predictions']].sum().reset_index()
            else:
                upper_df = upper_fitted[upper_fitted[id_column] == ts_id]
                lower_df = lower_fitted[lower_fitted[id_column] == ts_id]
            plt.fill_between(x=refined_df[date_column],
                             y1=upper_df['Predictions'],
                             y2=lower_df['Predictions'],
                             color='lightseagreen',
                             alpha=.1)
        if upper_predicted is not None and lower_predicted is not None:
            if level is not None:
                upper_df = upper_predicted.groupby(date_column)[['Predictions']].sum().reset_index()
                lower_df = lower_predicted.groupby(date_column)[['Predictions']].sum().reset_index()
            else:
                upper_df = upper_predicted[upper_predicted[id_column] == ts_id]
                lower_df = lower_predicted[lower_predicted[id_column] == ts_id]
            plt.fill_between(x=upper_df[date_column],
                             y1=upper_df['Predictions'],
                             y2=lower_df['Predictions'],
                             color='lightseagreen',
                             linestyle='dashed',
                             alpha=.25)

        plt.show()


