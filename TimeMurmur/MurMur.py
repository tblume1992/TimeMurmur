# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# from TimeMurmur.Optimizer import Optimize
from TimeMurmur.builder.Builder import Builder
from TimeMurmur.Model import Model
from TimeMurmur.utils.FeatureExtraction import get_features
from TimeMurmur.utils.utility_functions import infer_freq, smape, mape, mase, mse
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
            ma=None,
            fourier_order=10,
            seasonal_weights=None,
            weighted=True,
            n_basis=None,
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
            floor_bind=False,
            scale_type='standard',
            basis_difference=False,
            linear_test_window=None,
            seasonal_dummy=False,
            outlier_cap=None,
            ts_features=False,
            sample_weights=None,
            num_threads=1
            ):
        self.difference = difference
        self.scale = scale
        self.linear_trend = linear_trend
        self.id_column = id_column
        self.date_column = date_column
        self.target_column = target_column
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
                          ma=ma,
                          fourier_order=fourier_order,
                          freq=freq,
                          seasonal_weights=seasonal_weights,
                          weighted=weighted,
                          n_basis=n_basis,
                          seasonal_period=seasonal_period,
                          test_size=test_size,
                          linear_trend=linear_trend,
                          scale_type=scale_type,
                          basis_difference=basis_difference,
                          linear_test_window=linear_test_window,
                          seasonal_dummy=seasonal_dummy,
                          floor_bind=floor_bind,
                          floor=self.floor,
                          outlier_cap=outlier_cap,
                          ts_features=ts_features)
        print('Preprocessing Data: Scaling, building basis functions, capping outliers etc.')
        process_dataset = self.builder.preprocess(df)
        process_dataset = process_dataset.dropna(subset=['Murmur Target'])
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
        self.dataset = self.builder.build_dataset(process_dataset,
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
                      alpha=alpha,
                      num_threads=num_threads)
        self.model_obj = model.build_model()
        self.run_dict['global']['model'] = model.boosting_params
        if categorical_columns is None:
            cat_features = ['Murmur ID']
        else:
            cat_features = list(set(categorical_columns + ['Murmur ID']))
        if 'murmur_seasonal_dummy' in self.dataset.columns:
            cat_features += ['murmur_seasonal_dummy']
        self.run_dict['global']['cat_features'] = cat_features
        self.dataset = self.dataset.sort_values(by=['Murmur ID', date_column])
        self.train_X = self.dataset.drop(drop_columns+['Murmur Target', 'Murmur Data Split'], axis=1)
        eval_set = []
        train_y = self.dataset['Murmur Target']
        if labels is not None:
            train_y = labels
        if not eval_set:
            early_stopping_rounds = None
        self.columns = self.train_X.columns
        # self.train_X = dataset
        self.train_y = train_y
        self.model_obj.fit(self.train_X,
                           train_y, 
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds, 
                           categorical_feature=cat_features,
                           sample_weight=sample_weights)
        fitted = self.model_obj.predict(self.train_X)
        fitted_df = self.dataset[['Murmur ID', id_column, date_column, target_column, 'Murmur Target']]
        fitted_df['Predictions'] = fitted
        if self.scale or self.difference or self.linear_trend:
            fitted_df = self.unscale(fitted_df)
        if self.floor is not None:
            fitted_df['Predictions'] = fitted_df['Predictions'].clip(lower=self.floor)
        return fitted_df
    
    def ar_predict(self, forecast_horizon, pred_X):
        date_column = self.run_dict['global']['Date Column']
        id_column = 'Murmur ID'
        final_predicted = []
        self.pred_X = []
        print('Running Recursive Predictions')
        for i in tqdm(range(forecast_horizon)):
            refined_pred_X = pred_X.groupby('Murmur ID').nth(i)
            for _, dataset in self.run_dict['global']['AR Datasets'].items(): 
                refined_pred_X = refined_pred_X.merge(dataset,
                                                      on=['Murmur ID',
                                                          date_column],
                                                      how='left')
            self.refined_pred_X = refined_pred_X
            self.pred_X.append(refined_pred_X)
            model_X = refined_pred_X.drop([date_column], axis=1)
            predicted = self.model_obj.predict(model_X[list(self.columns)])
            predicted = pd.DataFrame(predicted, columns=['Predictions'])
            predicted[id_column] = refined_pred_X[id_column].values
            predicted[date_column] = refined_pred_X[date_column].values
            self.builder.build_future_ar_axis(predicted)
            final_predicted.append(predicted)
        self.pred_X = pd.concat(self.pred_X)
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
        if self.run_dict['global']['AR Datasets'] is not None:
            predicted_df = self.ar_predict(forecast_horizon,
                                        pred_X)
            predicted_df = predicted_df.merge(self.run_dict['global']['ID Mapping'],
                                              on='Murmur ID',
                                              how='left')
        else:
            self.pred_X = pred_X
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
        if self.scale or self.difference or self.linear_trend:
            predicted_df = self.unscale(predicted_df)
        if self.floor is not None:
            predicted_df['Predictions'] = predicted_df['Predictions'].clip(lower=self.floor)
        print('All Done!')
        return predicted_df

    def inverse_transform(self, df):
        scaler = self.run_dict['local'][df['Murmur ID'].iloc[0]]['scaler']
        df['Unscaled Predictions'] = df['Predictions']
        linear_trend = self.run_dict['local'][df['Murmur ID'].iloc[0]]['trend']
        if self.target_column in df.columns:
            if linear_trend:
                actuals = df['Murmur Target'].values
                linear = True
            else:
                actuals = df[self.target_column].values
                linear = False
            preds = scaler.inverse_transform(df['Predictions'].values.reshape(-1,1),
                                             actuals=actuals,
                                             linear=linear)
        else:
            preds = scaler.inverse_transform(df['Predictions'].values.reshape(-1,1))
        if any(np.isnan(preds)):
            print('Nan found when Inverse Transforming, use a more stable transformer such as "standard"')
        df['Predictions'] = preds
        return df
    
    def unscale(self, forecast_df):
        return forecast_df.groupby('Murmur ID').apply(self.inverse_transform)

    def explain_predictions(self, predicted_df):
        pred_X = self.pred_X[self.train_X.columns]
        self.explainer = shap.TreeExplainer(self.model_obj)
        shap_values = self.explainer.shap_values(pred_X)
        shap_values = pd.DataFrame(shap_values, columns=pred_X.columns)
    
        shap_values = shap_values.rename({'Murmur ID': 'Level'}, axis=1)
    
        shap_values[self.date_column] = self.pred_X[self.date_column].values
        shap_values['Murmur ID'] = self.pred_X['Murmur ID'].values
        shap_values = shap_values.merge(self.run_dict['global']['ID Mapping'],
                                        on='Murmur ID')
        # shap_values[self.date_column] = shap_values[self.date_column].dt.date
        shap_values['Predictions'] = predicted_df['Predictions']

        scale = (shap_values['Predictions'].values / shap_values.iloc[:, :-4].sum(axis=1).values)
        shap_values.iloc[:, :-4] = shap_values.iloc[:, :-4].mul(scale, axis=0)
        def rename_column(column_name, tags=None):
            if 'basis' in column_name:
                return 'Trend'
            if 'calc' in column_name:
                return 'TimeSeriesFeatures'
            if 'fourier' in column_name:
                return 'Seasonality'
            if 'ar_' in column_name:
                return 'LaggedData'
            if tags is not None:
                matching = [s for s in tags if column_name.split('_')[0] in s]
                if matching:
                    return matching
            return column_name
        shap_values.columns = [rename_column(i) for i in shap_values.columns]
        shap_values = shap_values.groupby(level=0, axis=1).sum()
        return shap_values

    def explain_fitted(self, fitted_df):
        pred_X = self.train_X
        self.explainer = shap.TreeExplainer(self.model_obj)
        shap_values = self.explainer.shap_values(pred_X)
        shap_values = pd.DataFrame(shap_values, columns=pred_X.columns)
    
        shap_values = shap_values.rename({'Murmur ID': 'Level'}, axis=1)
        shap_values[self.date_column] = self.dataset[self.date_column].values
        shap_values['Murmur ID'] = self.dataset['Murmur ID'].values
        shap_values = shap_values.merge(self.run_dict['global']['ID Mapping'],
                                        on='Murmur ID')
        # shap_values[self.date_column] = shap_values[self.date_column].dt.date
        shap_values['Predictions'] = fitted_df['Predictions'].values

        scale = (shap_values['Predictions'].values / shap_values.iloc[:, :-4].sum(axis=1).values)
        shap_values.iloc[:, :-4] = shap_values.iloc[:, :-4].mul(scale, axis=0)
        def rename_column(column_name, tags=None):
            if 'basis' in column_name:
                return 'Trend'
            if 'calc' in column_name:
                return 'TimeSeriesFeatures'
            if 'fourier' in column_name:
                return 'Seasonality'
            if 'ar_' in column_name:
                return 'LaggedData'
            if tags is not None:
                matching = [s for s in tags if column_name.split('_')[0] in s]
                if matching:
                    return matching
            return column_name
        shap_values.columns = [rename_column(i) for i in shap_values.columns]
        shap_values = shap_values.groupby(level=0, axis=1).sum()
        return shap_values

    def plot_explanations(self,
                          ts_id=None,
                          murmur_id=None,
                          level=None,
                          predicted_shap_vals=None,
                          fitted_shap_vals=None):
        if (murmur_id is None and ts_id is None) and level is None:
            raise ValueError('Must pass a level or time series ID')
        if (fitted_shap_vals is None and predicted_shap_vals is None):
            raise ValueError('Must pass a fitted or predicted shap val DF')
        date_column = self.run_dict['global']['Date Column']
        if murmur_id is not None:
            id_column = 'Murmur ID'
            ts_id = murmur_id
        else:
            id_column = self.run_dict['global']['ID Column']
        target_column = self.run_dict['global']['Target Column']
        if predicted_shap_vals is not None:
            plot_cols = [i for i in predicted_shap_vals.columns if i not in ['Predictions',
                                                                              'Murmur ID',
                                                                              self.id_column,
                                                                              self.date_column]]
        else:
            plot_cols = [i for i in fitted_shap_vals.columns if i not in ['Predictions',
                                                                              'Murmur ID',
                                                                              self.id_column,
                                                                              self.date_column]]
        if level == 'all':
            if fitted_shap_vals is not None:
                fitted_shap_vals = fitted_shap_vals.groupby(self.date_column)[plot_cols].sum().reset_index()
            if predicted_shap_vals is not None:
                predicted_shap_vals = predicted_shap_vals.groupby(self.date_column)[plot_cols].sum().reset_index()
        else:
            if fitted_shap_vals is not None:
                fitted_shap_vals = fitted_shap_vals[fitted_shap_vals[id_column] == ts_id]
            if predicted_shap_vals is not None:
                predicted_shap_vals = predicted_shap_vals[predicted_shap_vals[id_column] == ts_id]
        if predicted_shap_vals is not None and fitted_shap_vals is not None:
            vals = pd.concat([fitted_shap_vals, predicted_shap_vals])
            vals = vals.set_index(self.date_column)[plot_cols]
            vals.plot(kind='bar', stacked=True)
        elif fitted_shap_vals is not None:
            fitted_shap_vals = fitted_shap_vals.set_index(self.date_column)[plot_cols]
            self.fitted_shap_vals = fitted_shap_vals
            fitted_shap_vals.plot(kind='bar', stacked=True)
        elif predicted_shap_vals is not None:
            predicted_shap_vals = predicted_shap_vals.set_index(self.date_column)[plot_cols]
            predicted_shap_vals.plot(kind='bar', stacked=True)

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
        fig, ax = plt.subplots()
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

        ax.plot(refined_df[date_column],
                  refined_df['Predictions'],  
                  color='lightseagreen',
                  label='Fitted')
        if predicted is not None:
            if level is not None:
                predicted = predicted.groupby(date_column)[['Predictions']].sum().reset_index()
            else:
                predicted = predicted[predicted[id_column] == ts_id]
            ax.plot(predicted[date_column],
                      predicted['Predictions'],
                      color='lightseagreen',
                      linestyle='dashed',
                      label='Predicted')
        ax.plot(refined_df[date_column],
                  refined_df[target_column],
                  color='navy',
                  label='Target')
        if upper_fitted is not None and lower_fitted is not None:
            if level is not None:
                upper_df = upper_fitted.groupby(date_column)[[target_column,'Predictions']].sum().reset_index()
                lower_df = lower_fitted.groupby(date_column)[[target_column,'Predictions']].sum().reset_index()
            else:
                upper_df = upper_fitted[upper_fitted[id_column] == ts_id]
                lower_df = lower_fitted[lower_fitted[id_column] == ts_id]
            ax.fill_between(x=refined_df[date_column],
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
            ax.fill_between(x=upper_df[date_column],
                              y1=upper_df['Predictions'],
                              y2=lower_df['Predictions'],
                              color='lightseagreen',
                              linestyle='dashed',
                              alpha=.25)

        plt.legend()

        plt.show()

    def plot_importance(self, **kwargs):
        import lightgbm as gbm
        gbm.plot_importance(self.model_obj, **kwargs)


