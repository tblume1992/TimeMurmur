# # -*- coding: utf-8 -*-
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error
# import time
# from sklearn.model_selection import cross_val_score
# import numpy as np


# class Optimize:
#     def __init__(self,
#                  y,
#                  murmur_class,
#                  test_set,
#                  seasonal_period=0,
#                  n_folds=3,
#                  test_size=None,
#                  ):
#         self.y = y
#         self.murmur_class = murmur_class
#         self.test_set = test_set
#         self.seasonal_period = seasonal_period
#         self.n_folds = n_folds
#         self.test_size = test_size

#     def logic_layer(self):
#         n_samples = len(self.y)
#         test_size = n_samples//(self.n_folds + 1)
#         if n_samples - test_size < self.seasonal_period:
#             self.seasonal_period = 0

#     def get_space(self):
#         space = {
#             'n_basis': hp.choice('n_basis', np.arange(2, 20, 1)),
#             'fourier_order': scope.int(hp.quniform('fourier_order', 3, 20, 1)),
#             'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
#             'learning_rate': hp.quniform('learning_rate', 0.001, .1, .005),
#             'decay': hp.quniform('decay', 0.01, .99, .01),
#             # 'linear_trend': hp.choice("linear_trend", [True, False]),
#             'num_leaves': hp.choice('num_leaves', np.arange(8, 128, 2)),
#             # 'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
#         }
#         if self.seasonal_period:
#             space.update({'ar': hp.quniform('ar', 0, self.seasonal_period, 1)})
#         else:
#             space.update({'ar': hp.quniform('ar', 0, 4, 1)})
#         return space

#     def scorer(self, model_obj, y, metric, cv):
#         cv_splits = cv.split(y)
#         mses = []
#         for train_index, test_index in cv_splits:
#             try:
#                 # print(np.shape(y[train_index]))
#                 model_obj.fit(y[train_index])
#                 predicted = model_obj.predict(len(y[test_index]))
#                 mses.append(mean_squared_error(y[test_index], predicted))
#             except:
#                 mses.append(np.inf)
#         return_dict = {'loss': np.mean(mses),
#                        'eval_time': time.time(),
#                        'status': STATUS_OK}
#         return return_dict


#     def objective(self, params):
#         params = {
#             'n_basis': int(params['n_basis']),
#             'fourier_order': int(params['fourier_order']),
#             'ar': list(range(1, int(params['ar']) + 1)),
#             'n_estimators': int(params['n_estimators']),
#             # 'learning_rate': float(params['learning_rate']),
#             'decay': float(params['decay']),
#             # 'linear_trend': float(params['linear_trend']),
#             'num_leaves': int(params['num_leaves']),
#             # 'colsample_bytree': float(params['colsample_bytree']),
#         }
#         # print(params)
#         clf = self.murmur_class(**params)
#         # score = cross_val_score(clf, self.y, self.y, scoring=mean_squared_error, cv=TimeSeriesSplit(self.n_folds)).mean()
#         score = self.scorer(clf, self.y, mean_squared_error, self.test_set)
#         # print(f"MSE {score} params {params}")
#         return score

#     def fit(self):
#         # trials = Trials()
#         space = self.get_space()
#         best = fmin(fn=self.objective,
#                     space=space,
#                     algo=tpe.suggest,
#                     max_evals=500,
#                     # early_stop_fn=fn,
#                     verbose=False,
#                     return_argmin=False)
#         # print(best)
#         return best

# #%%
# # if __name__ == '__main__':
#     # optimizer = Optimize(np.array([1,2,3,4,5,6,7,8,9,10] *52), LazyProphet, 52, 5)
#     # optimized = optimizer.fit()
# #     splits = TimeSeriesSplit(2, test_size=52).split(y)
# #     for train_split, test_split in splits:
# #         train = y[train_split]
# #         test = y[test_split]


