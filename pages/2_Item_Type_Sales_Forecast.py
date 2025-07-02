# import streamlit as st
# import numpy as np
# import pandas as pd
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import plotly.graph_objects as go
# from utils import load_data

# st.set_page_config(page_title="Item Type Forecast", layout="wide")
# st.title("🔍 Forecast by Item Type")

# st.markdown("""
# This page allows you to **forecast sales for individual Item Types** over the next 12 months using a robust **SARIMAX time series model**.

# - 📦 **Item Types Included:** BEER, LIQUOR, WINE, NON-ALCOHOL
# - 📈 **Historical Data (blue line):** Shows actual sales per item type
# - 🔮 **Forecast (green line):** Predicted sales for the next 12 months
# - 🟩 **Confidence Interval (shaded area):** Indicates the range within which sales are expected to fall
# - 📅 **Hover Details:** View exact values and dates for better insights

# Select an Item Type from the dropdown above to view its sales forecast. This helps with **category-specific planning**, **inventory optimization**, and **targeted decision-making**.
# """)


# df = load_data()
# target_items = ['BEER', 'LIQUOR', 'WINE', 'NON-ALCOHOL']
# df_filtered = df[df['ITEM TYPE'].isin(target_items)]
# df_grouped = df_filtered.groupby(['DATE', 'ITEM TYPE'])['TOTAL SALES'].sum().unstack()

# # selected_item = st.selectbox("Select ITEM TYPE", target_items)
# selected_item = st.sidebar.selectbox("📦 Choose an Item Type to Forecast", target_items)


# series = df_grouped[selected_item]
# series_clean = series[series > 0].dropna()

# if series_clean.empty:
#     st.warning(f"No valid data available for {selected_item} after cleaning.")
# else:
#     log_series = np.log(series_clean)

#     try:
#         stepwise_model = auto_arima(log_series,
#                                     start_p=1, start_q=1,
#                                     max_p=3, max_q=3,
#                                     seasonal=True, m=12,
#                                     start_P=0, start_Q=0,
#                                     max_P=3, max_Q=3,
#                                     d=1, D=1,
#                                     trace=False,
#                                     error_action='ignore',
#                                     suppress_warnings=True,
#                                     stepwise=True)

#         model = SARIMAX(log_series,
#                         order=stepwise_model.order,
#                         seasonal_order=stepwise_model.seasonal_order,
#                         enforce_stationarity=False,
#                         enforce_invertibility=False)
#         results = model.fit(disp=False)

#         forecast = results.get_forecast(steps=12)
#         forecast_index = pd.date_range(start=log_series.index[-1] + pd.DateOffset(months=1),
#                                        periods=12, freq='MS')
#         forecast_values = np.exp(forecast.predicted_mean)
#         conf_int = np.exp(forecast.conf_int())

#         fig_item = go.Figure()
#         fig_item.add_trace(go.Scatter(x=series.index, y=series,
#                                       mode='lines', name='Historical', line=dict(color='blue'),hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))
#         fig_item.add_trace(go.Scatter(x=forecast_index, y=forecast_values,
#                                       mode='lines', name='Forecasted', line=dict(color='green'),hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))
#         fig_item.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0],
#                                       mode='lines', line=dict(color='lightgreen'), showlegend=False,hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))
#         fig_item.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1],
#                                       mode='lines', fill='tonexty',
#                                       fillcolor='rgba(144,238,144,0.3)',
#                                       line=dict(color='lightgreen'),
#                                       name='Confidence Interval', showlegend=False,hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

#         fig_item.update_layout(title=f"{selected_item} Sales Forecast",
#                                xaxis_title="Date", yaxis_title="Units Sold (in Thousands)",
#                                template="plotly_white", height=600, yaxis=dict(tickformat=".0s") )

#         st.plotly_chart(fig_item, use_container_width=True)

#         with st.expander("📋 Show Forecast Data Table & Explanation"):
#             st.markdown("""
#             #### ℹ️ Forecast Data Explanation:
#             - **Forecasted Sales:** Expected sales values for the selected item type over the next 12 months.
#             - **Lower Bound & Upper Bound:** Confidence range for the forecast — actual values are expected to lie within this interval.
#             - Use this data for **demand planning**, **inventory decisions**, and **seasonal strategy alignment**.
#             """)

#             forecast_df = pd.DataFrame({
#                 "Date": forecast_index,
#                 "Forecasted Sales": forecast_values,
#                 "Lower Bound": conf_int.iloc[:, 0],
#                 "Upper Bound": conf_int.iloc[:, 1]
#             })

#             st.dataframe(forecast_df, use_container_width=True)

            

#     except Exception as e:
#         st.error(f"Failed to forecast for {selected_item}: {str(e)}")


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import load_data, get_item_type_forecast

st.set_page_config(page_title="Item Type Forecast", layout="wide")
st.title("🔍 Forecast by Item Type")

st.markdown("""
This page allows you to **forecast sales for individual Item Types** over the next 12 months using a robust **SARIMAX time series model**.

- 📦 **Item Types Included:** BEER, LIQUOR, WINE, NON-ALCOHOL
- 📈 **Historical Data (blue line):** Shows actual sales per item type
- 🔮 **Forecast (green line):** Predicted sales for the next 12 months
- 🟩 **Confidence Interval (shaded area):** Indicates the range within which sales are expected to fall
- 📅 **Hover Details:** View exact values and dates for better insights

Select an Item Type from the dropdown above to view its sales forecast. This helps with **category-specific planning**, **inventory optimization**, and **targeted decision-making**.
""")

# Load data and define target items
df = load_data()
target_items = ['BEER', 'LIQUOR', 'WINE', 'NON-ALCOHOL']
selected_item = st.sidebar.selectbox("📦 Choose an Item Type to Forecast", target_items)

# Get pre-cached forecast data
series_clean, forecast_index, forecast_values, conf_int = get_item_type_forecast(selected_item)

# Handle empty data
if series_clean is None:
    st.warning(f"No valid data available for {selected_item} after cleaning.")
else:
    # Plotting
    fig_item = go.Figure()
    fig_item.add_trace(go.Scatter(x=series_clean.index, y=series_clean,
                                  mode='lines', name='Historical', line=dict(color='blue'),
                                  hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

    fig_item.add_trace(go.Scatter(x=forecast_index, y=forecast_values,
                                  mode='lines', name='Forecasted', line=dict(color='green'),
                                  hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

    fig_item.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0],
                                  mode='lines', line=dict(color='lightgreen'),
                                  showlegend=False, hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

    fig_item.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1],
                                  mode='lines', fill='tonexty',
                                  fillcolor='rgba(144,238,144,0.3)',
                                  line=dict(color='lightgreen'),
                                  name='Confidence Interval', showlegend=False,
                                  hovertemplate='%{x|%B %-d, %Y}, %{y:.2f}k'))

    fig_item.update_layout(
        title=f"{selected_item} Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Units Sold (in Thousands)",
        template="plotly_white",
        height=600,
        yaxis=dict(tickformat=".0s")
    )

    st.plotly_chart(fig_item, use_container_width=True)

    # Forecast Table & Explanation
    with st.expander("📋 Show Forecast Data Table & Explanation"):
        st.markdown("""
        #### ℹ️ Forecast Data Explanation:
        - **Forecasted Sales:** Expected sales values for the selected item type over the next 12 months.
        - **Lower Bound & Upper Bound:** Confidence range for the forecast — actual values are expected to lie within this interval.
        - Use this data for **demand planning**, **inventory decisions**, and **seasonal strategy alignment**.
        """)

        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecasted Sales": forecast_values,
            "Lower Bound": conf_int.iloc[:, 0],
            "Upper Bound": conf_int.iloc[:, 1]
        })

        st.dataframe(forecast_df, use_container_width=True)

