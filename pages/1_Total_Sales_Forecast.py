# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from utils import load_data

# st.set_page_config(page_title="Total Sales Forecast", layout="wide")
# st.title("📈 Overall Total Sales Forecast (12 months)")

# st.markdown("""
# This page provides a **12-month forecast** of overall total sales for core product types (`Wine`, `Beer`, `Liquor`, and `Non-Alcohol`) using time series modeling.

# - **Historical Data (cyan):** Actual sales in the past.
# - **Forecast (green):** Projected sales for the next 12 months.
# - **Confidence Interval (shaded):** Uncertainty range.
# - **Y-Axis:** Sales units (in thousands).
# """)

# # 🚀 Load and filter data for selected ITEM TYPEs only
# df = load_data()
# allowed_types = ['WINE', 'BEER', 'LIQUOR', 'NON-ALCOHOL']
# df = df[df['ITEM TYPE'].isin(allowed_types)]

# @st.cache_data
# def forecast_total_sales(df):
#     monthly_sales = df.groupby('DATE')['TOTAL SALES'].sum()
#     log_sales = np.log(monthly_sales)

#     stepwise_model = auto_arima(log_sales, start_p=1, start_q=1,
#                                 max_p=3, max_q=3,
#                                 seasonal=True, m=12,
#                                 start_P=0, start_Q=0,
#                                 max_P=3, max_Q=3,
#                                 d=1, D=1,
#                                 trace=False,
#                                 error_action='ignore',
#                                 suppress_warnings=True,
#                                 stepwise=True)

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

# # Forecasting
# monthly_total_sales, forecast_index, forecast_values, conf_int = forecast_total_sales(df)

# fig_total = go.Figure()

# fig_total.add_trace(go.Scatter(x=monthly_total_sales.index,y=monthly_total_sales.values,mode='lines',name='Historical',
#                                line=dict(color='#08FDD8'),hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

# fig_total.add_trace(go.Scatter(x=forecast_index,y=forecast_values,mode='lines',name='Forecast',
#                                line=dict(color='green'),hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

# fig_total.add_trace(go.Scatter(x=forecast_index,y=conf_int.iloc[:, 1],mode='lines',
#                                line=dict(color='lightgreen'),showlegend=False,hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

# fig_total.add_trace(go.Scatter(x=forecast_index,y=conf_int.iloc[:, 0],fill='tonexty',fillcolor='rgba(144,238,144,0.3)',mode='lines',line=dict(color='rgba(8,253,216,0.3)'),name='Confidence Interval',
#                                showlegend=False,hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'
# ))

# fig_total.update_layout(
#     title="Overall Forecast",
#     xaxis_title="Date",
#     yaxis_title="Units Sold (in Thousands)",
#     template='plotly_dark',
#     plot_bgcolor='#0E1117',
#     paper_bgcolor='#0E1117',
#     font=dict(color='white'),
#     legend=dict(bgcolor='rgba(0,0,0,0)'),
#     height=600,
#     yaxis=dict(tickformat=".0s")  # 👈 shows 60k, 80k, 100k on axis
# )





# st.plotly_chart(fig_total, use_container_width=True)

# with st.expander("📋 Show Forecast Data Table & Explanation"):

#     st.markdown("""
#     ### Forecast Overview
    
#     The table below shows the **forecasted total sales for the next 12 months** along with the corresponding **confidence intervals** (lower and upper bounds).
    
#     - **Date:** Month and year for each forecasted point.
#     - **Forecasted Sales:** Expected units sold (in thousands).
#     - **Lower Bound / Upper Bound:** Represents the 95% confidence interval for the forecast, giving a range where the actual sales are likely to fall.
    
#     Use this information for planning inventory, sales targets, and resource allocation.
#     """)

#     # Prepare forecast DataFrame with nicely formatted columns
#     forecast_df = pd.DataFrame({
#         "Date": forecast_index.strftime("%B %Y"),  # e.g., June 2021
#         "Forecasted Sales (k)": forecast_values.round(2),
#         "Lower Bound (k)": conf_int.iloc[:, 0].round(2),
#         "Upper Bound (k)": conf_int.iloc[:, 1].round(2)
#     })

#     # Display with some Streamlit formatting options
#     st.dataframe(forecast_df.style.format({
#         "Forecasted Sales (k)": "{:,.2f}",
#         "Lower Bound (k)": "{:,.2f}",
#         "Upper Bound (k)": "{:,.2f}"
#     }), use_container_width=True)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import load_data, prepare_forecast


st.set_page_config(page_title="Total Sales Forecast", layout="wide")
st.title("📈 Overall Total Sales Forecast (12 months)")

st.markdown("""
This page provides a **12-month forecast** of overall total sales for core product types (`Wine`, `Beer`, `Liquor`, and `Non-Alcohol`) using time series modeling.

- **Historical Data (cyan):** Actual sales in the past.
- **Forecast (green):** Projected sales for the next 12 months.
- **Confidence Interval (shaded):** Uncertainty range.
- **Y-Axis:** Sales units (in thousands).
""")

from utils import load_data, prepare_forecast

# Load and filter data
df = load_data()
allowed_types = ['WINE', 'BEER', 'LIQUOR', 'NON-ALCOHOL']
df = df[df['ITEM TYPE'].isin(allowed_types)]

# Cache the forecasting step
@st.cache_data
def get_forecast(df_json):
    df = pd.read_json(df_json)
    return prepare_forecast(df)

df_json = df.to_json()
monthly_total_sales, forecast_index, forecast_values, conf_int = get_forecast(df_json)

# 📊 Plotting
fig_total = go.Figure()

fig_total.add_trace(go.Scatter(x=monthly_total_sales.index, y=monthly_total_sales.values, mode='lines', name='Historical',
                               line=dict(color='#08FDD8'), hovertemplate='%{x|%B %Y}, %{y:.2f}k'))

fig_total.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecast',
                               line=dict(color='green'), hovertemplate='%{x|%B %Y}, %{y:.2f}k'))

fig_total.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], mode='lines',
                               line=dict(color='lightgreen'), showlegend=False, hovertemplate='%{x|%B %Y}, %{y:.2f}k'))

fig_total.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], fill='tonexty',
                               fillcolor='rgba(144,238,144,0.3)', mode='lines',
                               line=dict(color='rgba(8,253,216,0.3)'), name='Confidence Interval',
                               showlegend=False, hovertemplate='%{x|%B %Y}, %{y:.2f}k'))

fig_total.update_layout(
    title="Overall Forecast",
    xaxis_title="Date",
    yaxis_title="Units Sold (in Thousands)",
    template='plotly_dark',
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='white'),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    height=600,
    yaxis=dict(tickformat=".0s")
)

st.plotly_chart(fig_total, use_container_width=True)

# 📋 Forecast Table
with st.expander("📋 Show Forecast Data Table & Explanation"):
    st.markdown("""
    ### Forecast Overview

    The table below shows the **forecasted total sales for the next 12 months** along with the corresponding **confidence intervals** (lower and upper bounds).

    - **Date:** Month and year for each forecasted point.
    - **Forecasted Sales:** Expected units sold (in thousands).
    - **Lower Bound / Upper Bound:** Represents the 95% confidence interval for the forecast, giving a range where the actual sales are likely to fall.

    Use this information for planning inventory, sales targets, and resource allocation.
    """)

    forecast_df = pd.DataFrame({
        "Date": forecast_index.strftime("%B %Y"),
        "Forecasted Sales (k)": forecast_values.round(2),
        "Lower Bound (k)": conf_int.iloc[:, 0].round(2),
        "Upper Bound (k)": conf_int.iloc[:, 1].round(2)
    })

    st.dataframe(forecast_df.style.format({
        "Forecasted Sales (k)": "{:,.2f}",
        "Lower Bound (k)": "{:,.2f}",
        "Upper Bound (k)": "{:,.2f}"
    }), use_container_width=True)
