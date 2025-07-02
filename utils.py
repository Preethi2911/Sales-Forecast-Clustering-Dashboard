# import pandas as pd

# def load_data():
#     df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
#     df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
#     df['TOTAL SALES'] = df[['RETAIL SALES']].sum(axis=1)
#     return df
#---------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# def load_data():
#     df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
#     df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
#     df['TOTAL SALES'] = df[['RETAIL SALES']].sum(axis=1)
#     return df

# def prepare_forecast(df):
#     monthly_sales = df.groupby('DATE')['TOTAL SALES'].sum()
#     log_sales = np.log(monthly_sales)

#     stepwise_model = auto_arima(log_sales, start_p=1, start_q=1,
#                                  max_p=3, max_q=3,
#                                  seasonal=True, m=12,
#                                  start_P=0, start_Q=0,
#                                  max_P=3, max_Q=3,
#                                  d=1, D=1,
#                                  trace=False,
#                                  error_action='ignore',
#                                  suppress_warnings=True,
#                                  stepwise=True,
#                                  lazy=True)

#     model = SARIMAX(log_sales,
#                     order=stepwise_model.order,
#                     seasonal_order=stepwise_model.seasonal_order)
#     results = model.fit(disp=False)

#     forecast = results.get_forecast(steps=12)
#     forecast_index = pd.date_range(start=log_sales.index[-1] + pd.DateOffset(months=1),
#                                    periods=12, freq='MS')
#     forecast_values = np.exp(forecast.predicted_mean)
#     conf_int = np.exp(forecast.conf_int())

#     return monthly_sales, forecast_index, forecast_values, conf_int

#----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_data():
    df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    df['TOTAL SALES'] = df[['RETAIL SALES']].sum(axis=1)
    return df

def prepare_forecast(df):
    monthly_sales = df.groupby('DATE')['TOTAL SALES'].sum()
    log_sales = np.log(monthly_sales)

    stepwise_model = auto_arima(log_sales, start_p=1, start_q=1,
                                 max_p=3, max_q=3,
                                 seasonal=True, m=12,
                                 start_P=0, start_Q=0,
                                 max_P=3, max_Q=3,
                                 d=1, D=1,
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True,
                                 lazy=True)

    model = SARIMAX(log_sales,
                    order=stepwise_model.order,
                    seasonal_order=stepwise_model.seasonal_order)
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=12)
    forecast_index = pd.date_range(start=log_sales.index[-1] + pd.DateOffset(months=1),
                                   periods=12, freq='MS')
    forecast_values = np.exp(forecast.predicted_mean)
    conf_int = np.exp(forecast.conf_int())

    return monthly_sales, forecast_index, forecast_values, conf_int

# === ⏱️ NEW: Cached forecast function for item types ===
from functools import lru_cache

@lru_cache(maxsize=10)
def get_item_type_forecast(item_type: str):
    df = load_data()
    df_filtered = df[df['ITEM TYPE'] == item_type]
    df_grouped = df_filtered.groupby('DATE')['TOTAL SALES'].sum()
    series_clean = df_grouped[df_grouped > 0].dropna()

    if series_clean.empty:
        return None, None, None, None

    log_series = np.log(series_clean)

    stepwise_model = auto_arima(log_series,
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                seasonal=True, m=12,
                                start_P=0, start_Q=0,
                                max_P=3, max_Q=3,
                                d=1, D=1,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    model = SARIMAX(log_series,
                    order=stepwise_model.order,
                    seasonal_order=stepwise_model.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=12)
    forecast_index = pd.date_range(start=log_series.index[-1] + pd.DateOffset(months=1),
                                   periods=12, freq='MS')
    forecast_values = np.exp(forecast.predicted_mean)
    conf_int = np.exp(forecast.conf_int())

    return series_clean, forecast_index, forecast_values, conf_int


