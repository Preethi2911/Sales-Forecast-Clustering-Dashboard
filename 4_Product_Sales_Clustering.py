import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from neo4j import GraphDatabase, TRUST_ALL_CERTIFICATES
from pyvis.network import Network
import streamlit.components.v1 as components
 
st.set_page_config(page_title="Product Sales Clustering", layout="wide")
st.title("📊 Clustering Products by Sales")

st.markdown("""
### 🔍 About This Dashboard

This interactive clustering dashboard segments products based on their **total sales volume** using the clustering algorithm.  
You can:

- 📦 **Select an Item Type** (e.g., Wine, Beer, Liquor)
- 📅 **Choose Year and Month** to view relevant data
- 🎯 **Adjust the number of clusters** (2–6 groups)
- 📊 **Visualize** how products group together based on sales
- 🔎 **Explore items** within each sales-based group

This helps identify **low-performing** and **top-performing** products at a glance and can guide inventory and marketing decisions.
""")


# # Apply Streamlit CSS to ensure iframe transparency
# st.markdown("""
# <style>
#     iframe {
#         background: transparent !important;
#         border: none !important;
#     }
#     .stApp {
#         background: #222222 !important;
#     }
# </style>
# """, unsafe_allow_html=True)
 
# === Initialize Session State ===
if 'file_counter' not in st.session_state:
    st.session_state.file_counter = 0
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = None
 
# === Load CSV ===
file_path = "Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)
 
# === Data Cleaning ===
df['RETAIL SALES'] = df['RETAIL SALES'].fillna(0)
df['WAREHOUSE SALES'] = df['WAREHOUSE SALES'].fillna(0)
df = df[(df['RETAIL SALES'] >= 0) & (df['WAREHOUSE SALES'] >= 0)]
df['TOTAL SALES'] = df['RETAIL SALES']  # Set TOTAL SALES to RETAIL SALES only
 
# === Dropdown for Item Type ===
item_types = ['WINE', 'BEER', 'LIQUOR','NON-ALCOHOL']
selected_item_type = st.sidebar.selectbox("🛒 Select Item Type", item_types, key="item_type")
 
# === Filter Data by Item Type ===
item_df_all = df[df['ITEM TYPE'].str.upper() == selected_item_type].copy()
valid_years = sorted([year for year in item_df_all['YEAR'].unique() if year in [2021, 2022, 2023, 2024]])
 
if not valid_years:
    st.warning(f"No data available for item type '{selected_item_type}'. Please select another item type.")
else:
    # === Year Selection ===
    col1, col2 = st.columns(2)
    with col1:
        # Reset year if it’s invalid for the new item type
        if st.session_state.selected_year not in valid_years:
            st.session_state.selected_year = valid_years[0] if valid_years else None
        selected_year = st.sidebar.selectbox("📅 Select Year", valid_years,
                                     index=valid_years.index(st.session_state.selected_year) if st.session_state.selected_year in valid_years else 0,
                                     key="year_select")
        st.session_state.selected_year = selected_year
 
    # === Month Selection ===
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_map = {month: idx + 1 for idx, month in enumerate(months)}
 
    valid_months = item_df_all[item_df_all['YEAR'] == selected_year]['MONTH'].unique()
    valid_month_names = sorted([month for month, num in month_map.items() if num in valid_months],
                              key=lambda x: month_map[x])
 
    if not valid_month_names:
        st.warning(f"No data available for {selected_item_type} in {selected_year}. Please select another year or item type.")
    else:
        with col2:
            # Reset month if it’s invalid for the new year/item type
            if st.session_state.selected_month not in valid_month_names:
                st.session_state.selected_month = valid_month_names[0] if valid_month_names else None
            selected_month = st.sidebar.selectbox("📆 Select Month", valid_month_names,
                                         index=valid_month_names.index(st.session_state.selected_month) if st.session_state.selected_month in valid_month_names else 0,
                                         key="month_select")
            st.session_state.selected_month = selected_month
 
        # === Filter Data ===
        item_df = item_df_all[
            (item_df_all['YEAR'] == selected_year) &
            (item_df_all['MONTH'] == month_map[selected_month])
        ].copy()
 
        item_grouped = item_df.groupby(['ITEM CODE', 'ITEM DESCRIPTION'])[['TOTAL SALES']].sum().reset_index()
 
        if item_grouped.empty:
            st.warning(f"No items of type '{selected_item_type}' found for {selected_month} {selected_year}.")
        else:
            # === Clustering ===
            max_clusters = min(6, len(item_grouped))  # Limit clusters to number of samples
            if max_clusters < 2:
                st.warning(f"Only {len(item_grouped)} item(s) found for {selected_item_type} in {selected_month} {selected_year}. Clustering requires at least 2 items.")
            else:
                n_clusters = st.sidebar.slider("🔧 Select number of groups", 2, max_clusters, min(3, max_clusters))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                item_grouped['CLUSTER'] = kmeans.fit_predict(item_grouped[['TOTAL SALES']])
 
                # === Check for Single Cluster ===
                unique_clusters = item_grouped['CLUSTER'].nunique()
                if unique_clusters <= 1:
                    st.warning(f"Insufficient data variation for clustering {selected_item_type} in {selected_month} {selected_year}. All items are in a single cluster.")
                else:
                    cluster_means = item_grouped.groupby('CLUSTER')['TOTAL SALES'].mean().sort_values().reset_index()
                    ordered_cluster_ids = cluster_means['CLUSTER'].tolist()
                    cluster_id_to_rank = {cid: idx for idx, cid in enumerate(ordered_cluster_ids)}
                    item_grouped['CLUSTER_RANK'] = item_grouped['CLUSTER'].map(cluster_id_to_rank)
 
                    name_map = {
                        2: {0: "Low Sales", 1: "High Sales"},
                        3: {0: "Low Sales", 1: "Medium Sales", 2: "High Sales"},
                        4: {0: "Very Low Sales", 1: "Low Sales", 2: "Medium Sales", 3: "High Sales"},
                        5: {0: "Very Low Sales", 1: "Low Sales", 2: "Medium Sales", 3: "High Sales", 4: "Very High Sales"},
                        6: {0: "Very Low Sales", 1: "Low Sales", 2: "Medium Low Sales", 3: "Medium Sales", 4: "High Sales", 5: "Very High Sales"},
                    }.get(n_clusters, {})
 
                    item_grouped['GROUP'] = item_grouped['CLUSTER_RANK'].map(name_map)
 
                    cluster_counts = item_grouped['GROUP'].value_counts().reset_index()
                    cluster_counts.columns = ['Group', f'Number of {selected_item_type}']
 
                    st.subheader(f"{selected_item_type} Product Group Distribution for {selected_month} {selected_year}")
                    fig = px.pie(cluster_counts, names='Group', values=f'Number of {selected_item_type}',
                                 title=f'{selected_item_type} Clusters Based on Total Sales ({selected_month} {selected_year})',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig)
                    st.markdown(f"""
                                ### 🔎 Explore {selected_item_type} Items by Group for {selected_month} {selected_year}

                                Below, you can **select a specific sales group** to see which products fall under it.  
                                Each group represents items that perform similarly in terms of **total sales**.  
                                This breakdown helps in:

                                - 🧭 Identifying underperforming vs. high-performing products  
                                - 🛍️ Understanding which products dominate in each cluster  
                                - 📦 Making targeted decisions on stock, promotions, or discontinuation
                                """)

                    # === Neo4j connection info ===
                    NEO4J_URI = "bolt://f6629d9e.databases.neo4j.io"
                    NEO4J_USER = "neo4j"
                    NEO4J_PASSWORD = "3_iuBF7umrFxlzfWB7uLT0yyvH24xSizWaTESJoyFbY"
 
                    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), encrypted=True, trust=TRUST_ALL_CERTIFICATES)
 
                    def create_graph(tx, items, cluster_name, year, month, item_type, position):
                        for _, row in items.iterrows():
                            tx.run("""
                                MERGE (p:Product {code: $code, group: $group, year: $year, month: $month, item_type: $item_type, position: $position})
                                SET p.description = $desc, p.sales = $sales
                            """, code=row['ITEM CODE'], desc=row['ITEM DESCRIPTION'], sales=row['TOTAL SALES'],
                                 group=cluster_name, year=year, month=month, item_type=item_type, position=position)
 
                    with driver.session() as session:
                        session.run("MATCH (p:Product) DETACH DELETE p")
 
                        for group in item_grouped['GROUP'].unique():
                            top_items = item_grouped[item_grouped['GROUP'] == group].nlargest(10, 'TOTAL SALES')
                            session.execute_write(create_graph, top_items, group, selected_year, month_map[selected_month], selected_item_type, "top")
 
                            bottom_items = item_grouped[item_grouped['GROUP'] == group].nsmallest(10, 'TOTAL SALES')
                            session.execute_write(create_graph, bottom_items, group, selected_year, month_map[selected_month], selected_item_type, "bottom")
 
                    def get_top_products_graph(cluster_name):
                        with driver.session() as session:
                            result = session.run("""
                                MATCH (p:Product)
                                WHERE p.group = $group AND p.year = $year AND p.month = $month AND p.item_type = $item_type AND p.position = 'top'
                                RETURN p.code AS code, p.description AS desc, p.sales AS sales, p.group AS group
                                ORDER BY p.sales DESC
                                LIMIT 10
                            """, group=cluster_name, year=selected_year, month=month_map[selected_month], item_type=selected_item_type)
                            return result.data()
 
                    def get_bottom_products_graph(cluster_name):
                        with driver.session() as session:
                            result = session.run("""
                                MATCH (p:Product)
                                WHERE p.group = $group AND p.year = $year AND p.month = $month AND p.item_type = $item_type AND p.position = 'bottom'
                                RETURN p.code AS code, p.description AS desc, p.sales AS sales, p.group AS group
                                ORDER BY p.sales ASC
                                LIMIT 10
                            """, group=cluster_name, year=selected_year, month=month_map[selected_month], item_type=selected_item_type)
                            return result.data()
 
                    def visualize_products(data, group, position):
                        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
                        net.force_atlas_2based()
 
                        cluster_colors = {}
                        color_palette = ['#FF5733', '#33C3FF', '#8D33FF', '#33FF57', '#FFC300', '#E91E63', '#9C27B0']
                        color_idx = 0
                        clusters = set([item['group'] for item in data if item['group']])
 
                        for cluster in clusters:
                            cluster_colors[cluster] = color_palette[color_idx % len(color_palette)]
                            color_idx += 1
 
                        center_label = f"{position} 10: {group}"
                        net.add_node(center_label, label=center_label, shape='ellipse', color="#4682B4",
                                     font={'size': 20, 'color': 'white'})
 
                        for item in data:
                            tooltip = (
                                f"""Item Code: {item['code']}
                                    Description: {item['desc']}
                                    Total Sales: {item['sales']:.2f}
                                    Group: {item['group']}"""
                                                                )
 
                            net.add_node(item['code'], label=item['desc'], title=tooltip,
                                         shape='dot', color="#FF6347", size=25,
                                         font={'size': 16, 'color': 'white'})
 
                            net.add_edge(center_label, item['code'], color='gray')
 
                        net.set_options("""
                        {
                            "nodes": {
                                "font": {"size": 16, "face": "Arial", "color": "white"},
                                "scaling": {"label": {"enabled": true}}
                            },
                            "edges": {
                                "color": {"color": "gray"}
                            },
                            "physics": {
                                "forceAtlas2Based": {
                                    "gravitationalConstant": -120,
                                    "springLength": 100
                                },
                                "minVelocity": 0.75,
                                "solver": "forceAtlas2Based"
                            },
                            "configure": {
                                "enabled": false
                            },
                            "layout": {
                                "improvedLayout": true
                            },
                            "interaction": {
                                "zoomView": true,
                                "dragView": true
                            }
                        }
                        """)
 
                        # Create custom HTML wrapper
                        # st.session_state.file_counter += 1
                        # file_name = f"{position.lower()}_network_{group.replace(' ', '_')}_{st.session_state.file_counter}.html"
                        file_name = "Product_clusters_filtered.html"
                        net.save_graph(file_name)
 
                        # Read Pyvis-generated HTML and wrap it in a custom div
                        with open(file_name, 'r') as file:
                            pyvis_html = file.read()
 
                        custom_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <style>
                                body, html, #canvas, .vis-network, div {{
                                    background: transparent !important;
                                    margin: 0 !important;
                                    padding: 0 !important;
                                    overflow: hidden !important;
                                    width: 100%;
                                    height: 600px;
                                }}
                            </style>
                        </head>
                        <body>
                            <div style="background: transparent; overflow: hidden; width: 100%; height: 600px;">
                                {pyvis_html}
                            </div>
                        </body>
                        </html>
                        """
 
                        with open(file_name, 'w') as file:
                            file.write(custom_html)
 
                        return file_name
 
                    st.subheader(f"🕸️ Product Networks and Details for {selected_month} {selected_year}")
                    selected_group = st.selectbox("Select a group to view top/bottom networks and items", sorted(item_grouped['GROUP'].unique()))
 
                    st.subheader(f"Top 10 Products in '{selected_group}' Group")
                    top_data = get_top_products_graph(selected_group)
                    top_path = visualize_products(top_data, selected_group, "Top")
                    components.html(open(top_path, 'r').read(), height=600, scrolling=False)
 
                    st.subheader(f"Bottom 10 Products in '{selected_group}' Group")
                    bottom_data = get_bottom_products_graph(selected_group)
                    bottom_path = visualize_products(bottom_data, selected_group, "Bottom")
                    components.html(open(bottom_path, 'r').read(), height=600, scrolling=False)
 
                    # === Toggle Table Button ===
                    if 'show_table' not in st.session_state:
                        st.session_state.show_table = False
 
                    if st.button("Show All Items" if not st.session_state.show_table else "Hide Table"):
                        st.session_state.show_table = not st.session_state.show_table
 
                    if st.session_state.show_table:
                        st.subheader(f"{selected_item_type} Items in '{selected_group}' Group")
                        st.dataframe(item_grouped[item_grouped['GROUP'] == selected_group][['ITEM CODE', 'ITEM DESCRIPTION', 'TOTAL SALES']])