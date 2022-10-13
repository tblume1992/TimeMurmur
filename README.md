# TimeMurmur
Requires the forecast period is the same for all time series.
## Quickstart
```
pip install TimeMurmur
```
Get example dataset and ensure the date column is Datetime:
```
from TimeMurmur.utils.utility_functions import get_data
train_df = get_data()
train_df['Datetime'] = pd.to_datetime(train_df['Datetime'])
```
This dataset is a subset of the weekly data from M4, it includes the required ID, Datetime, and History columns.

The most basic model you can fit is:
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W')
predicted = murmur.predict(20)
```
These parameters are required, the freq is 'W' for Weekly following standard frequency nomenclature.
You can take a look with the helper function:
```
for i in range(3):
    murmur.plot(fitted, predicted=predicted, murmur_id=i)
```
## Adding AR Lags
In order to do recursive forecasting utilizing past values, just pass a list of lags you want to the ar parameter:
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W',
                    n_basis=[10],
                    ar=[1,2,3,4])
predicted = murmur.predict(20)
```
Here we use a linear basis function with 4 ar lags passed as a list.
## Adding Seasonality
Seasonality works similar to the ar lags. We can pass multiple seasonal periods in a list if we have multiple seasonalities to account for.
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W',
                    n_basis=[10],
                    ar=[1,2,3,4],
                    seassonal_period=[4,52])
predicted = murmur.predict(20)
```
## LightGBM Parameters
You can pass a few of the most influential LightGBM parameters to fit such as `num_iterations` and `num_leaves`:
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W',
                    n_basis=[10],
                    ar=[1,2,3,4],
                    seassonal_period=[4,52],
                    num_iterations=100,
                    learning_rate=.1,
                    num_leaves=31)
predicted = murmur.predict(20)
```
## Fitting with category exogenous.
This is a 'ID' axis variable since it never changes across time only across IDs. Since it is a string we pass it to 'categorical_columns' as well.
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W',
                    n_basis=[10],
                    ar=[1,2,3,4],
                    seassonal_period=[4,52],
                    num_iterations=100,
                    learning_rate=.1,
                    num_leaves=31,
                    categorical_columns=['category'],
                    id_feature_columns=['category'])
predicted = murmur.predict(20)
```
## Fitting Quantiles
To fit quantiles pass `quantile` for `objective` and the desired quantile percentage to `alpha`.
```
murmur = Murmur(floor=0)
fitted = murmur.fit(train_df,
                    target_column='History',
                    id_column='ID',
                    date_column='Datetime',
                    freq='W',
                    n_basis=[10],
                    ar=[1,2,3,4],
                    seassonal_period=[4,52],
                    num_iterations=100,
                    learning_rate=.1,
                    num_leaves=31,
                    objective='quantile',
                    alpha=.9)
predicted = murmur.predict(20)
```
