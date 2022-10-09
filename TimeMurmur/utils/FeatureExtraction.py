import numpy as np
import pandas as pd
import statsmodels.api as sm
import inspect
import itertools
from scipy.stats import entropy
from ThymeBoost import ThymeBoost as tb


class FeatureExtraction:
    def __init__(self, series, period):
        self.series = pd.Series(np.reshape(series, (-1, ))).reset_index(drop = True)
        if len(self.series) < 12:
            self.period = 0
        else:
            self.period = period
        # self.extract()
        return
    
    def scale_data(self):
        series_min = np.min(self.series)
        series_max = np.max(self.series)
        series_mean = np.mean(self.series)
        self.series = (self.series - series_mean)/(series_max - series_min)
        
    # def get_differenced_series(self):
    #     self.diff_series = self.series.diff().dropna()
    #     self.diff_2_series = self.diff_series.diff().dropna()
    #     self.seasonal_diff_series = self.diff_series.diff(self.period).dropna()
    #     return
    
    # def get_acf(self):
    #     self.acf = sm.tsa.stattools.acf(self.series, nlags = 10)
    #     self.diff_acf = sm.tsa.stattools.acf(self.diff_series, nlags = 10)
    #     self.diff_2_acf = sm.tsa.stattools.acf(self.diff_2_series, nlags = 10)
    #     self.seasonal_acf = sm.tsa.stattools.acf(self.seasonal_diff_series)
    #     return

    def get_tiles(self):
        try:
            self.tiled = [self.series[i:i+self.n_season] for i in \
                          range(0, len(self.series), self.n_season)]
        except:
            self.tiled = [self.series[i:i+2] for i in \
                          range(0, len(self.series), 2)]
        return
    
    # def get_pacf(self):
    #     self.pacf = sm.tsa.stattools.pacf(self.series, nlags = 10)
    #     self.diff_pacf = sm.tsa.stattools.pacf(self.diff_series, nlags = 10)
    #     self.diff_2_pacf = sm.tsa.stattools.pacf(self.diff_2_series, nlags = 10)
    #     self.seasonal_pacf = sm.tsa.stattools.pacf(self.seasonal_diff_series)
        
    #     return
    
    def get_lazy_prophet_output(self):
        self.lp_model = tb.ThymeBoost(verbose=0,)
        self.lp_output = self.lp_model.fit(self.series,
                                           trend_estimator='linear',
                                           seasonal_estimator='fourier',
                                           seasonal_period=self.period,

                                           )
        # self.lp_coefs = self.lp_model.coefs[0]
        self.lp_deseasonalized = self.lp_output['y'] - self.lp_output['seasonality']
        self.lp_remainder = self.lp_output['y'] - self.lp_output['yhat']
        self.lp_detrended = self.lp_output['y'] - self.lp_output['trend']
        return
        
    
    # def calc_hurst(self):
    #     series = np.array(self.series)
    #     lags = range(2, 100)
    #     tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    #     poly = np.polyfit(np.log(lags), np.log(tau), 1)

    #     return poly[0]*2.0
       
    def calc_entropy(self):
        return entropy(self.series + 10, base = 2)
    
    # def calc_acf_10(self):
    #     return np.sum(self.acf[1:11]**2)
    
    # def calc_acf_diff_10(self):
    #     return np.sum(self.diff_acf[1:11]**2)
    
    # def calc_acf(self):
    #     return self.acf[1]
    
    # def calc_acf_diff(self):
    #     return self.diff_acf[1]
    
    # def calc_acf_2_diff_10(self):
    #     return np.sum(self.diff_2_acf[1:11]**2)
    
    # def calc_acf_2_diff(self):
    #     return self.diff_2_acf[1]
    
    # def calc_seasonal_acf(self):
    #     return self.seasonal_acf[1]
    
    # #ToDo
    # #def calc_flat_spots(self):
        
    
    # def calc_pacf_10(self):
    #     return np.sum(self.pacf[1:11]**2)
    
    # def calc_pacf_diff_10(self):
    #     return np.sum(self.diff_pacf[1:11]**2)
    
    # def calc_pacf(self):
    #     return self.pacf[1]
    
    # def calc_pacf_diff(self):
    #     return self.diff_pacf[1]
    
    # def calc_pacf_2_diff_10(self):
    #     return np.sum(self.diff_2_pacf[1:11]**2)
    
    # def calc_pacf_2_diff(self):
    #     return self.diff_2_pacf[1]
    
    # def calc_seasonal_pacf(self):
    #     return self.seasonal_pacf[1]
    
    def calc_crossing_points(self):
        crossing_points = len(list(itertools.groupby(self.series-np.median(self.series),
                                                     lambda Input: Input > 0))) 
        return crossing_points - 1

    def count_zeros(self):
        self.zero_count = sum(self.series == 0)
        return

    def calc_n_zeros(self):
        return self.zero_count

    def calc_seasonal_period(self):
        return self.period
    
    # def calc_lp_curvature(self):
    #     return self.lp_coefs[1]
    
    # def calc_lp_linearity(self):
    #     return self.lp_coefs[0]
    
    def calc_std(self):
        return np.std(self.series)
    
    def calc_trend_strength(self):
        lp_trend = np.std(self.lp_remainder)**2 / np.std(self.lp_deseasonalized)**2
        
        return max((0, 1 - lp_trend))
    
    def calc_lumpiness(self):
        return np.var([np.var(i) for i in self.tiled])
    
    def calc_stability(self):
        return np.var([np.mean(i) for i in self.tiled])
    
    def calc_n_seasons(self):
        return self.n_season
    
    def calc_seasonal_strength(self):
        return max(0, 1-(np.var(self.lp_remainder)/np.var(self.lp_detrended)))
    
    def calc_series_length(self):
        return len(self.series)
        
    def calc_adf(self):
        return sm.tsa.stattools.adfuller(self.series, maxlag = 1)[0]

    def calc_kpss(self):
        return sm.tsa.stattools.kpss(self.series, lags = 1)[0]
    
    def calc_mean_abs_change(self):
        return np.mean(np.abs(np.diff(self.series)))
    
    def calc_mean_gradient(self):
        return np.mean(np.gradient(self.series))
    
    def calc_std_gradient(self):
        return np.std(np.gradient(self.series))
    
    def calc_mean_median_diff(self):
        return np.mean(self.series) - np.median(self.series)

    def calc_skewness(self):
        return self.series.skew()
    
    def calc_kurtosis(self):
        return self.series.kurtosis()
    
    def calc_reoccurring_perc(self):
        return len(self.series.unique())/len(self.series)
    
    #from tsfresh, thank you!
    @staticmethod
    def _roll(a, shift):
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        idx = shift % len(a)
        return np.concatenate([a[-idx:], a[:-idx]])
    
    def calc_number_peaks(self):
        n = 3
        x = np.asarray(self.series)
        x_reduced = x[n:-n]
    
        res = None
        for i in range(1, n + 1):
            result_first = (x_reduced > FeatureExtraction._roll(x, i)[n:-n])
    
            if res is None:
                res = result_first
            else:
                res &= result_first
    
            res &= (x_reduced > FeatureExtraction._roll(x, -i)[n:-n])
        return np.sum(res)
    #Too time consuming, vectorized implementation seems wonky
# =============================================================================
#     def calc_lp_spike(self):
#         leave_one_out = [self.lp_remainder.drop(i).var() for i in range(len(self.lp_remainder))]
#         
#         return np.var(leave_one_out)
# =============================================================================
    
    def extract(self):
        self.features = {}
        self.count_zeros()
        self.scale_data()
        # self.get_differenced_series()
        if self.period:
            self.n_season = int(len(self.series)/self.period)
        else:
            self.n_season = 0
        # self.get_acf()
        # self.get_pacf()
        self.get_tiles()
        try:
            self.get_lazy_prophet_output()
        except:
            self.lp_deseasonalized = np.zeros(len(self.series))
            self.lp_remainder = np.zeros(len(self.series))
            self.lp_detrended = np.zeros(len(self.series))
        for method in inspect.getmembers(FeatureExtraction, predicate=inspect.isfunction):
            if 'calc' in method[0]:
                try:
                    self.features[method[0]] = method[1](self)
                except:
                    self.features[method[0]] = 0
        return self.features
