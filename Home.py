import streamlit as st

st.set_page_config(page_title="Sales Forecast & Clustering Dashboard", layout="wide")

st.title("📊 Welcome to the Sales Forecast & Clustering Dashboard")

st.markdown("""
Welcome to your interactive **Sales Forecasting & Clustering** platform!  
This dashboard provides powerful tools to gain **data-driven insights** into your product sales and supplier performance.

### 🔍 Explore the Features:

- 📈 **Total Sales Forecast**  
  View a 12-month forecast of overall sales trends powered by advanced time series models.

- 🧾 **Item Type Forecasting**  
  Analyze detailed sales forecasts for individual product categories like Beer, Liquor, Wine, and more.

- 🏢 **Supplier Clustering**  
  Cluster suppliers based on **Total Retail Sales** and **Unique Products** to identify key partners and optimize strategies.

- 📦 **Product Sales Clustering**  
  Segment products by sales volume into groups, helping spot top and low performers for smarter inventory and marketing.

---

Use the **sidebar menu** to navigate between modules.  
Each section includes interactive charts, tables, and explanations to support your business decisions.
""")
