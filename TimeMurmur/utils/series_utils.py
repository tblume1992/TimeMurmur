# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

def linear_test(y, seasonal_period, linear_trend):
    y = y.copy().reshape((-1,))
    xi = np.arange(1, len(y) + 1)
    # xi = xi**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
    trend_line = slope*xi*r_value + intercept
    if seasonal_period is not None:
        required_len = 1.5 * max(seasonal_period)
    else:
        required_len = 6
    if linear_trend and len(y) > required_len:
        n_bins = (1 + len(y)**(1/3) * 2)
        splitted_array = np.array_split(y.reshape(-1,), int(n_bins))
        mean_splits = np.array([np.mean(i) for i in splitted_array])
        asc_array = np.sort(mean_splits)
        desc_array = np.flip(asc_array)
        if all(asc_array == mean_splits):
            growth = True
        elif all(desc_array == mean_splits):
            growth = True
        else:
            growth = False
        if (r_value > .9 and growth):
            linear = True
        else:
            linear = False
    else:
        linear = False
    slope = slope * r_value
    return trend_line, linear, slope, intercept, r_value