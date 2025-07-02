import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
 
st.set_page_config(page_title="Supplier Clustering Dashboard", layout="wide")
 
# Title and description
st.title("Supplier Clustering Dashboard")
st.markdown("""
This dashboard clusters suppliers based on **Total Retail Sales** and **Number of Unique Products**.
Use the sidebar to adjust filters and explore supplier clusters. Hover over points for details, click legend items to toggle clusters, or zoom/pan to explore.
The supplier-cluster network graph shows the top 15 suppliers per cluster, and the supplier-product graph below it dynamically updates to show only the selected cluster's suppliers and their top products.
""")
 
# Initialize session state for toggling table visibility and search term
if 'show_table' not in st.session_state:
    st.session_state.show_table = False
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""
 
# Cache Neo4j driver to reuse connection
@st.cache_resource
def get_neo4j_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))
 
# Cache Neo4j data upload
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def cached_push_to_neo4j(uri, user, password, suppliers_df, products_df):
    driver = get_neo4j_driver(uri, user, password)
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
       
        # Create cluster nodes
        for cluster in suppliers_df['Cluster_Label'].unique():
            session.run(
                "CREATE (c:Cluster {label: $label})",
                label=cluster
            )
       
        # Create supplier nodes and relationships
        for _, row in suppliers_df.iterrows():
            session.run(
                """
                CREATE (s:Supplier {name: $name, total_sales: $total_sales, unique_products: $unique_products})
                WITH s
                MATCH (c:Cluster {label: $cluster_label})
                CREATE (s)-[:IN_CLUSTER]->(c)
                """,
                name=row['SUPPLIER'],
                total_sales=float(row['RETAIL SALES']),
                unique_products=int(row['UNIQUE PRODUCTS']),
                cluster_label=row['Cluster_Label']
            )
       
        # Create product nodes and relationships
        for _, row in products_df.iterrows():
            session.run(
                """
                MATCH (s:Supplier {name: $supplier})
                CREATE (p:Product {name: $product, sales: $sales})
                CREATE (s)-[:SELLS]->(p)
                """,
                supplier=row['SUPPLIER'],
                product=row['ITEM DESCRIPTION'],
                sales=float(row['RETAIL SALES'])
            )
    return True
 
# Cache supplier-cluster data fetching
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def fetch_supplier_cluster_data(uri, user, password, _suppliers_df, _products_df):
    driver = get_neo4j_driver(uri, user, password)
    with driver.session() as session:
        results = session.run("""
            MATCH (s:Supplier)-[:IN_CLUSTER]->(c:Cluster)
            RETURN s.name AS supplier, s.total_sales AS total_sales,
                   s.unique_products AS unique_products, c.label AS cluster
        """)
        return pd.DataFrame([r.data() for r in results])
 
# Cache supplier-product data fetching
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def fetch_supplier_product_data(uri, user, password, selected_cluster, _suppliers_df, _products_df):
    driver = get_neo4j_driver(uri, user, password)
    with driver.session() as session:
        if selected_cluster == 'All':
            query = """
                MATCH (s:Supplier)-[:SELLS]->(p:Product)
                RETURN s.name AS supplier, s.total_sales AS total_sales,
                       s.unique_products AS unique_products, p.name AS product, p.sales AS product_sales
            """
        else:
            query = """
                MATCH (s:Supplier)-[:IN_CLUSTER]->(c:Cluster {label: $cluster})
                MATCH (s)-[:SELLS]->(p:Product)
                RETURN s.name AS supplier, s.total_sales AS total_sales,
                       s.unique_products AS unique_products, p.name AS product, p.sales AS product_sales
            """
        results = session.run(query, cluster=selected_cluster)
        return pd.DataFrame([r.data() for r in results])
 
# Load dataset
try:
    df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
except FileNotFoundError:
    st.error("Error: 'Warehouse_and_Retail_Sales.csv' not found. Please ensure the file exists.")
    st.stop()

item_types = ['WINE', 'BEER', 'LIQUOR','NON-ALCOHOL'] 
# Sidebar filters
st.sidebar.header("Filters")
item_type = st.sidebar.selectbox("Item Type", ['All'] + sorted(item_types), key="item_type")
year = st.sidebar.selectbox("Year", ['All'] + sorted(df['YEAR'].dropna().astype(int).unique().tolist()), index=0)
# month = st.sidebar.selectbox("Month", ['All'] + sorted(df['MONTH'].dropna().astype(int).unique().tolist()), index=0)

# Filter data for months based on item_type and year selection
month_filter_df = df.copy()
if item_type != 'All':
    month_filter_df = month_filter_df[month_filter_df['ITEM TYPE'] == item_type]
if year != 'All':
    month_filter_df = month_filter_df[month_filter_df['YEAR'] == int(year)]

# Now get the available months from filtered data
available_months = sorted(month_filter_df['MONTH'].dropna().astype(int).unique().tolist())
month = st.sidebar.selectbox("Month", ['All'] + available_months, index=0)

num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=5, value=3, step=1)
# exclude_negatives = st.sidebar.checkbox("Exclude Negative Retail Sales", value=False)
 
# Filter dataset
filtered_df = df.copy()
if item_type != 'All':
    filtered_df = filtered_df[filtered_df['ITEM TYPE'] == item_type]
if year != 'All':
    filtered_df = filtered_df[filtered_df['YEAR'] == int(year)]
if month != 'All':
    filtered_df = filtered_df[filtered_df['MONTH'] == int(month)]
# if exclude_negatives:
#     filtered_df = filtered_df[filtered_df['RETAIL SALES'] >= 0]
 
# Aggregate supplier metrics
try:
    supplier_df = filtered_df.groupby('SUPPLIER').agg({
        'RETAIL SALES': 'sum',
        'ITEM DESCRIPTION': 'nunique'
    }).reset_index().rename(columns={'ITEM DESCRIPTION': 'UNIQUE PRODUCTS'})
except KeyError as e:
    st.error(f"Error: Missing required column(s): {e}")
    st.stop()
 
# Check if sufficient data exists
if len(supplier_df) < num_clusters:
    st.error(f"Not enough suppliers ({len(supplier_df)}) for {num_clusters} clusters. Adjust filters or reduce clusters.")
    st.stop()
if len(supplier_df) == 0:
    st.error("No suppliers match the selected filters. Try selecting 'All' for Item Type, Year, or Month.")
    st.stop()
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(supplier_df[['RETAIL SALES', 'UNIQUE PRODUCTS']])
 
# Perform clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
supplier_df['Cluster'] = kmeans.fit_predict(X_scaled)
 
# Define cluster labels
cluster_labels = {
    2: {0: 'Low Volume', 1: 'High Volume'},
    3: {0: 'Small-Scale Specialists', 1: 'Mid-Tier Balanced Suppliers', 2: 'High-Volume Generalists'},
    4: {0: 'Niche Low-Volume Vendors', 1: 'Focused Mid-Performers', 2: 'Diversified Mid-Volume Vendors', 3: 'Large-Scale Comprehensive Suppliers'},
    5: {0: 'Micro-Niche Vendors', 1: 'Focused Low-Mid Volume Suppliers', 2: 'Balanced Moderate Suppliers', 3: 'High Variety Niche Players', 4: 'Top-Tier Full-Line Suppliers'}
}[num_clusters]
supplier_df['Cluster_Label'] = supplier_df['Cluster'].map(cluster_labels)
 
# Filter top 15 suppliers per cluster, excluding grouping column in apply
top_suppliers_df = supplier_df.groupby('Cluster').apply(
    lambda x: x.nlargest(15, 'RETAIL SALES'), include_groups=False
).reset_index().drop(columns=['level_1'])
 
# Aggregate top 2 products for top suppliers
top_suppliers_list = top_suppliers_df['SUPPLIER'].tolist()
top_products_df = filtered_df[filtered_df['SUPPLIER'].isin(top_suppliers_list)]
top_products_df = top_products_df.groupby(['SUPPLIER', 'ITEM DESCRIPTION'])['RETAIL SALES'].sum().reset_index()
top_products_df = top_products_df.sort_values(['SUPPLIER', 'RETAIL SALES'], ascending=[True, False])
top_products_df = top_products_df.groupby('SUPPLIER').head(2).reset_index(drop=True)
 
# Create scatter plot
title_parts = [item_type if item_type != 'All' else None, f"Month {month}" if month != 'All' else None, str(year) if year != 'All' else None]
title_parts = [part for part in title_parts if part]
title_suffix = ', '.join(title_parts) if title_parts else 'All Items'
plot_title = f"Supplier Clusters by Retail Sales and Product Variety ({title_suffix})"
 
fig = px.scatter(
    supplier_df,
    x='RETAIL SALES',
    y='UNIQUE PRODUCTS',
    color='Cluster_Label',
    size='RETAIL SALES',
    hover_data=['SUPPLIER', 'RETAIL SALES', 'UNIQUE PRODUCTS'],
    color_discrete_sequence=['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080'][:num_clusters],
    size_max=40
)
 
# Add annotations for top 3 suppliers per cluster
for cluster in range(num_clusters):
    top_suppliers = supplier_df[supplier_df['Cluster'] == cluster].nlargest(3, 'RETAIL SALES')
    for _, row in top_suppliers.iterrows():
        fig.add_annotation(
        x=row['RETAIL SALES'],
        y=row['UNIQUE PRODUCTS'],
        text=row['SUPPLIER'],
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-20,
        font=dict(size=10, color='white'),
        bgcolor="rgba(0, 0, 0, 0.7)",
        opacity=0.9,
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )

 
fig.update_layout(
    title={
        'text': plot_title,
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20, 'color': 'white'}
    },
    xaxis_title="Total Retail Sales (Unit)",
    yaxis_title="Number of Products Under Supplier",
    legend={
        'title': 'Supplier Clusters',
        'x': 1.05,
        'y': 1,
        'font': {'size': 12, 'color': 'white'},
        'bgcolor': 'rgba(0, 0, 0, 0.5)',
        'bordercolor': 'white',
        'borderwidth': 1
    },
    showlegend=True,
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='white'),
    xaxis=dict(
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=1,
        linecolor='white',
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    ),
    yaxis=dict(
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=1,
        linecolor='white',
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    ),
    margin=dict(l=50, r=150, t=80, b=50),
    hovermode='closest',
    height=600
)

 
st.plotly_chart(fig, use_container_width=True)
 
# Button to toggle clustering data table
if st.button("Show/Hide Clustering Data Table"):
    st.session_state.show_table = not st.session_state.show_table
    if not st.session_state.show_table:
        st.session_state.search_term = ""  # Clear search term when hiding table
 
# Display table if toggled on
if st.session_state.show_table:
    st.subheader("Clustering Data")
    st.markdown("Search for suppliers and download the filtered clustering data as a CSV file.")
   
    # Search functionality
    st.session_state.search_term = st.text_input(
        "Search Supplier",
        value=st.session_state.search_term,
        placeholder="Enter supplier name...",
        key=f"search_supplier_{st.session_state.show_table}"
    )
    display_df = supplier_df.copy()
    if st.session_state.search_term:
        display_df = display_df[display_df['SUPPLIER'].str.contains(st.session_state.search_term, case=False, na=False)]
   
    # Display searchable table or no-results message
    if display_df.empty and st.session_state.search_term:
        st.warning("No suppliers found matching the search term.")
    else:
        st.dataframe(display_df, use_container_width=True, height=400)
   
    # Download button for filtered CSV
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Clustering Data as CSV",
        data=csv,
        file_name="supplier_clustering_data.csv",
        mime="text/csv"
    )
 
# Display cluster summary
st.header("Cluster Summary")
cluster_summary = top_suppliers_df.groupby('Cluster_Label').agg({
    'RETAIL SALES': ['mean', 'sum'],
    'UNIQUE PRODUCTS': 'mean',
    'SUPPLIER': ['count', lambda x: ', '.join(x.head(3).tolist())]
}).round(2)
cluster_summary.columns = ['Retail Sales (Mean)', 'Retail Sales (Total)', 'Unique Products (Mean)', 'Supplier Count', 'Top Suppliers']
st.dataframe(cluster_summary, use_container_width=True)
 
# Display cluster interpretation
st.header("Cluster Interpretation")
st.markdown({
    2: """
    - **Low Volume**: Suppliers with lower sales and fewer products. Suitable for niche inventory.
    - **High Volume**: Suppliers with higher sales and more products. Ideal for bulk orders or partnerships.
    """,
    3: """
    - **Small-Scale Specialists**: Low sales, few products. Niche suppliers for specialized offerings (e.g., craft liquors).
    - **Mid-Tier Balanced Suppliers**: Moderate sales and variety. Reliable for consistent inventory needs.
    - **High-Volume Generalists**: High sales, many products. Major distributors for partnerships or promotions.
    """,
    4: """
    - **Niche Low-Volume Vendors**: Very low sales, limited variety. Ideal for exclusive products.
    - **Focused Mid-Performers**: Moderate sales, focused range. Suitable for targeted inventory.
    - **Diversified Mid-Volume Vendors**: Moderate sales, broader variety. Good for balanced strategies.
    - **Large-Scale Comprehensive Suppliers**: High sales, extensive variety. Key for large-scale operations.
    """,
    5: """
    - **Micro-Niche Vendors**: Minimal sales, few products. Highly specialized for exclusive offerings.
    - **Focused Low-Mid Volume Suppliers**: Low to moderate sales, limited variety. Suitable for specific categories.
    - **Balanced Moderate Suppliers**: Moderate sales and variety. Reliable for consistent supply.
    - **High Variety Niche Players**: Moderate sales, high variety. Good for diverse inventory.
    - **Top-Tier Full-Line Suppliers**: High sales, extensive range. Strategic for large-scale operations.
    """
}[num_clusters])
 
# Neo4j visualization
st.header("Supplier-Cluster Network Graph ")
st.markdown("Visualizes the top 15 suppliers per cluster. Select a cluster below to filter the supplier-product graph.")
 
try:
    # Neo4j connection details
    uri = "neo4j+ssc://44ceff47.databases.neo4j.io:7687"
    user = "neo4j"
    password = "q6gS4Kz0oi794hw6LdqrgptzsO014rXgBDO9wiXTeW0"
   
    # Upload data to Neo4j (cached)
    cached_push_to_neo4j(uri, user, password, top_suppliers_df, top_products_df)
   
    # Fetch supplier-cluster data (cached)
    vis_data = fetch_supplier_cluster_data(uri, user, password, top_suppliers_df, top_products_df)
   
    # Build supplier-cluster network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    cluster_colors = {cluster: color for cluster, color in zip(vis_data['cluster'].unique(), ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080'][:num_clusters])}
   
    for cluster in vis_data['cluster'].unique():
        net.add_node(cluster, label=cluster, color=cluster_colors[cluster], shape='dot', size=25)
   
    for _, row in vis_data.iterrows():
        supplier = row['supplier']
        cluster = row['cluster']
        hover = f"Supplier: {supplier}\nTotal Sales Unit: {row['total_sales']}\nUnique Products: {row['unique_products']}"
        net.add_node(supplier, label=supplier, color="#AAAAAA", size=15, title=hover)
        net.add_edge(supplier, cluster)
   
    net.save_graph("supplier_clusters_filtered.html")


    with open("supplier_clusters_filtered.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        '<body>',
        '<body style="margin: 0; overflow: hidden; background-color: #000000;">'
    )
    components.html(html_content, height=600, scrolling=False)
    
   
    # Supplier-Product Network with Cluster Filter
    st.header("Supplier-Top Products Network Graph ")
    st.markdown("Shows the top 15 suppliers and their top two products for the selected cluster.")
   
    # Cluster selection dropdown
    selected_cluster = st.sidebar.selectbox("Select Cluster for Supplier-Product Graph", ['All'] + sorted(vis_data['cluster'].unique().tolist()), index=0)
   
    # Fetch supplier-product data (cached)
    product_vis_data = fetch_supplier_product_data(uri, user, password, selected_cluster, top_suppliers_df, top_products_df)
   
    # Build supplier-product network
    product_net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    for _, row in product_vis_data.iterrows():
        supplier = row['supplier']
        product = row['product']
        hover_supplier = f"Supplier: {supplier}\nTotal Sales Unit: {row['total_sales']}\nUnique Products: {row['unique_products']}"
        hover_product = f"Product: {product}\nSales: {row['product_sales']}"
        product_net.add_node(supplier, label=supplier, color="#FF6347", size=20, title=hover_supplier)
        product_net.add_node(product, label=product, color="#4682B4", size=10, title=hover_product)
        product_net.add_edge(supplier, product)
   
    product_net.save_graph("supplier_products_filtered.html")
    with open("supplier_products_filtered.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        '<body>',
        '<body style="margin: 0; overflow: hidden; background-color: #000000;">'
    )
    components.html(html_content, height=600, scrolling=False)
 
except Exception as e:
    st.error(f"Failed to connect to Neo4j or visualize data: {e}")




























# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import plotly.express as px
# import streamlit as st
 
# st.set_page_config(page_title="Supplier Clustering Dashboard", layout="wide")
 
# # Title and description
# st.title("Supplier Clustering Dashboard")
# st.markdown("""
# This dashboard clusters suppliers based on their **Total Retail Sales** and **Number of Unique Products**.
# Use the sidebar to select item type, year, month, number of clusters, or exclude negative sales.
# Hover over points to view supplier details, click legend items to toggle clusters, or zoom/pan to explore.
# The clusters help identify key suppliers for partnerships, inventory optimization, or marketing strategies.
# """)
 
# # Load the dataset
# try:
#     df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
# except FileNotFoundError:
#     st.error("Error: 'Warehouse_and_Retail_Sales.csv' not found. Please ensure the file is in the same directory.")
#     st.stop()
 
# # Sidebar for controls
# st.sidebar.header("Filters")
# item_type = st.sidebar.selectbox(
#     "Item Type",
#     options=['All'] + sorted(df['ITEM TYPE'].dropna().unique().tolist()),
#     index=0
# )
# year = st.sidebar.selectbox(
#     "Year",
#     options=['All'] + sorted(df['YEAR'].dropna().astype(int).unique().tolist()),
#     index=0
# )
# month = st.sidebar.selectbox(
#     "Month",
#     options=['All'] + sorted(df['MONTH'].dropna().astype(int).unique().tolist()),
#     index=0
# )
# num_clusters = st.sidebar.slider(
#     "Number of Clusters",
#     min_value=2,
#     max_value=5,
#     value=3,
#     step=1
# )
# exclude_negatives = st.sidebar.checkbox("Exclude Negative Retail Sales", value=False)
 
# # Filter the dataset
# filtered_df = df.copy()
# if item_type != 'All':
#     filtered_df = filtered_df[filtered_df['ITEM TYPE'] == item_type]
# if year != 'All':
#     filtered_df = filtered_df[filtered_df['YEAR'] == int(year)]
# if month != 'All':
#     filtered_df = filtered_df[filtered_df['MONTH'] == int(month)]
# if exclude_negatives:
#     filtered_df = filtered_df[filtered_df['RETAIL SALES'] >= 0]
 
# # Step 1: Aggregate supplier metrics
# try:
#     supplier_df = filtered_df.groupby('SUPPLIER').agg({
#         'RETAIL SALES': 'sum',
#         'ITEM DESCRIPTION': 'nunique'
#     }).reset_index()
# except KeyError as e:
#     st.error(f"Error: Missing required column(s): {e}")
#     st.stop()
 
# # Step 2: Prepare features
# features = ['RETAIL SALES', 'ITEM DESCRIPTION']
# supplier_df = supplier_df.dropna(subset=features)
 
# # Check if there are enough suppliers for clustering
# if len(supplier_df) < num_clusters:
#     st.error(f"Not enough suppliers ({len(supplier_df)}) for {num_clusters} clusters. Please reduce the number of clusters or adjust filters.")
#     st.stop()
# if len(supplier_df) == 0:
#     st.error("No suppliers match the selected filters. Please adjust the filters (e.g., select 'All' for Item Type, Year, or Month).")
#     st.stop()
 
# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(supplier_df[features])
 
# # Step 3: Cluster
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)
 
# # Step 4: Add cluster labels to DataFrame
# supplier_df['Cluster'] = clusters
 
# # Step 5: Define custom cluster labels (dynamic based on number of clusters)
# cluster_labels = {}
# for i in range(num_clusters):
#     if num_clusters == 3:
#         cluster_labels[i] = {
#             0: 'Small-Scale Specialists\n(low sales, few products)',
#             1: 'Mid-Tier Balanced Suppliers\n(moderate sales, moderate variety)',
#             2: 'High-Volume Generalists\n(high sales, many products)'
#         }.get(i, f'Cluster {i}')
#     else:
#         cluster_labels[i] = f'Cluster {i}\n(sales and variety vary)'
# supplier_df['Cluster_Label'] = supplier_df['Cluster'].map(cluster_labels)
 
# # Step 6: Create interactive Plotly plot
# fig = px.scatter(
#     supplier_df,
#     x='RETAIL SALES',
#     y='ITEM DESCRIPTION',
#     color='Cluster_Label',
#     size='RETAIL SALES',
#     hover_data=['SUPPLIER', 'RETAIL SALES', 'ITEM DESCRIPTION'],
#     color_discrete_sequence=['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080'][:num_clusters],  # Red, Blue, Green, Orange, Purple
#     size_max=40
# )
 
# # Add annotations for top 3 suppliers per cluster
# for cluster in range(num_clusters):
#     top_suppliers = supplier_df[supplier_df['Cluster'] == cluster].nlargest(3, 'RETAIL SALES')
#     for _, row in top_suppliers.iterrows():
#         fig.add_annotation(
#             x=row['RETAIL SALES'],
#             y=row['ITEM DESCRIPTION'],
#             text=row['SUPPLIER'],
#             showarrow=True,
#             arrowhead=1,
#             ax=20,
#             ay=-20,
#             font=dict(size=10, color='white'),  # <-- change text color to white
#             bgcolor="rgba(0,0,0,0.7)",          # <-- dark semi-transparent background for readability
#             opacity=0.9,
#             bordercolor="white",                # <-- white border for contrast
#             borderwidth=1,
#             borderpad=4
#         )

# # Construct dynamic plot title based on filters
# title_parts = []
# if item_type != 'All':
#     title_parts.append(item_type)
# if month != 'All':
#     title_parts.append(f"Month {month}")
# if year != 'All':
#     title_parts.append(str(year))
# title_suffix = ', '.join(title_parts) if title_parts else 'All Items'
# plot_title = f"Supplier Clusters by Retail Sales and Product Variety ({title_suffix})"
 
# # Update layout for neatness
# fig.update_layout(
#     title={
#         'text': plot_title,
#         'y': 0.95,
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top',
#         'font': {'size': 20, 'color': 'white'}
#     },
#     xaxis_title="Total Retail Sales ($)",
#     yaxis_title="Number of Unique Products",
#     legend={
#         'title': 'Supplier Clusters',
#         'x': 1.05,
#         'y': 1,
#         'font': {'size': 12, 'color': 'white'},
#         'bgcolor': 'rgba(0, 0, 0, 0.5)',
#         'bordercolor': 'white',
#         'borderwidth': 1
#     },
#     showlegend=True,
#     plot_bgcolor='#0E1117',
#     paper_bgcolor='#0E1117',
#     font=dict(color='white'),
#     xaxis=dict(
#         gridcolor='rgba(255, 255, 255, 0.1)',
#         showline=True,
#         linewidth=1,
#         linecolor='white',
#         zerolinecolor='rgba(255, 255, 255, 0.2)'
#     ),
#     yaxis=dict(
#         gridcolor='rgba(255, 255, 255, 0.1)',
#         showline=True,
#         linewidth=1,
#         linecolor='white',
#         zerolinecolor='rgba(255, 255, 255, 0.2)'
#     ),
#     margin=dict(l=50, r=150, t=80, b=50),
#     hovermode='closest',
#     height=600
# )

 
# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)
 
# # Step 7: Display cluster summary
# st.header("Cluster Summary")
# cluster_summary = supplier_df.groupby('Cluster').agg({
#     'RETAIL SALES': ['mean', 'sum'],
#     'ITEM DESCRIPTION': 'mean',
#     'SUPPLIER': ['count', lambda x: ', '.join(x.head(3).tolist())]
# }).round(2)
 
# cluster_summary.columns = ['Retail Sales (Mean)', 'Retail Sales (Total)', 'Unique Products (Mean)', 'Supplier Count', 'Top Suppliers']
# st.dataframe(cluster_summary, use_container_width=True)
 
# # Step 8: Display business-friendly interpretation
# st.header("Cluster Interpretation")
# if num_clusters == 3:
#     st.markdown("""
#     - **Cluster 0: Small-Scale Specialists**  
#       Low retail sales, fewer unique products. Likely niche suppliers with specialized offerings (e.g., craft liquors or unique wines). Consider for high-margin or exclusive products.
#     - **Cluster 1: Mid-Tier Balanced Suppliers**  
#       Moderate retail sales and product variety. Reliable for consistent inventory needs. Good for stable, mid-range products.
#     - **Cluster 2: High-Volume Generalists**  
#       High retail sales and many unique products. Major distributors with broad portfolios (e.g., Crown Imports). Prioritize for partnerships, bulk orders, or promotions.
#     """)
# else:
#     st.markdown(f"""
#     Clusters are dynamically generated based on your selection of {num_clusters} clusters.
#     - Lower-numbered clusters (e.g., Cluster 0) typically have lower sales and fewer products.
#     - Higher-numbered clusters (e.g., Cluster {num_clusters-1}) typically have higher sales and more products.
#     - Review the summary table to understand each cluster's characteristics and top suppliers.
#     """)

