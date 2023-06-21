# -*- coding: utf-8 -*-
import warnings
import numpy as np
import lightgbm as gbm
from catboost import Pool, CatBoostRegressor
# import optuna.integration.lightgbm as gbm
warnings.filterwarnings("ignore")


class Model:

    def __init__(self,
                 objective='regression',
                 metric='rmse',
                 learning_rate=.1,
                 min_child_samples=5,
                 num_leaves=50,
                 num_iterations=50,
                 return_proba=False,
                 boosting_params=None,
                 scale_pos_weights=None,
                 is_unbalance=True,
                 alpha=None,
                 num_threads=1):
        self.objective = objective
        self.return_proba = return_proba
        if boosting_params is None:
            self.boosting_params = {
                                    "objective": objective,
                                    "metric": metric,
                                    # "tweedie_variance_power": 1.7,
                                    "verbosity": 1,
                                    "boosting_type": "gbdt",
                                    "seed": 42,
                                    'linear_tree': False,
                                    'learning_rate': learning_rate,
                                    'min_child_samples': min_child_samples,
                                    'num_leaves': num_leaves,
                                    'num_iterations': num_iterations,
                                    'alpha': alpha,
                                    'scale_pos_weights': scale_pos_weights,
                                    'is_unbalance': is_unbalance,
                                    'num_threads': num_threads
                                }
        else:
            self.boosted_params = boosting_params
            
    def build_model(self):
        if self.objective == 'regression' or self.objective == 'quantile' or self.objective == 'mape':
            model_obj = gbm.LGBMRegressor(**self.boosting_params)
        if self.objective == 'binary':
            model_obj = gbm.LGBMClassifier(**self.boosting_params)
        return model_obj
    
    def build_dataset(self,
                      dataset,
                      categorical_features=None,
                      test_size=None):
        split = 'Murmur Data Split'
        if test_size is None:
            validation_set = None
            eval_set = []
            train_set = dataset.drop(split, axis=1)
        else:       
            validation_set = dataset[dataset[split] == 'Test'].drop(split, axis=1)
            train_set = dataset[dataset[split] == 'Train'].drop(split, axis=1)
            test_X = validation_set.drop('Murmur Target', axis=1)
            test_y = validation_set['Murmur Target']
            eval_set = [(test_X, test_y)]
        train_X = train_set.drop('Murmur Target', axis=1)
        train_y = train_set['Murmur Target']

        del train_set
        if validation_set is not None:
            del validation_set
        return train_X, train_y, eval_set

    
