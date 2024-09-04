import pandas as pd
import matplotlib.pyplot as plt 
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")

# notes
# Text Elements: st.title(), st.header(), st.subheader(), st.write(), st.markdown()
# Data Display: st.dataframe(), st.table(), st.json()
# Charts and Plots: st.line_chart(), st.area_chart(), st.bar_chart(), st.pyplot(), st.altair_chart(), etc.
# Widgets: st.button(), st.radio(), st.selectbox(), st.slider(), st.checkbox(), st.file_uploader(), etc.
# Layouts: st.sidebar (for adding elements to a sidebar), st.columns() (for multi-column layouts), st.expander() (for collapsible sections)


st.title('Analisis Pelean')

@st.cache_data
def load_data():
    # Load and preprocess data here
    df = pd.read_csv('C:/Users/orlin/Downloads/Analisis Pelean/analysis_data.csv')
    # df_terima = list(data.filter(like='Total Terima').columns)

    df['Terima S1 2023'] = df['Total Terima Januari 2023'] + \
    df['Total Terima Februari 2023'] +\
    df['Total Terima Maret 2023'] +\
    df['Total Terima April 2023'] +\
    df['Total Terima Mei 2023'] +\
    df['Total Terima Juni 2023']

    df['Terima S2 2023'] = df['Total Terima Juli 2023'] +\
    df['Total Terima Agustus 2023'] +\
    df['Total Terima September 2023'] +\
    df['Total Terima Oktober 2023'] +\
    df['Total Terima November 2023'] +\
    df['Total Terima Desember 2023']

    df['Terima S1 2024'] = df['Total Terima Januari 2024'] +\
    df['Total Terima Februari 2024'] +\
    df['Total Terima Maret 2024'] +\
    df['Total Terima April 2024'] +\
    df['Total Terima Mei 2024'] +\
    df['Total Terima Juni 2024'] 
    return df

df = load_data()

# Define the options for the selection box
opsi_distrik = ['Show All'] + list(df['DISTRIK'].unique())

# Create the selection box
selected_distrik_opsi = st.selectbox('Choose a district:', opsi_distrik)

# Filter the data based on the selected district
if selected_distrik_opsi == 'Show All':
    filtered_data = df
else:
    filtered_data = df[df['DISTRIK'] == selected_distrik_opsi]

# Create the stacked bar chart using Plotly
st.subheader(f'Stacked Bar Chart for District: {selected_distrik_opsi}')
fig = go.Figure()

# Add traces for each category
fig.add_trace(go.Bar(x=filtered_data['DISTRIK'], y=filtered_data['Terima S1 2023'], name='Terima S1 2023'))
fig.add_trace(go.Bar(x=filtered_data['DISTRIK'], y=filtered_data['Terima S2 2023'], name='Terima S2 2023'))
fig.add_trace(go.Bar(x=filtered_data['DISTRIK'], y=filtered_data['Terima S1 2024'], name='Terima S1 2024'))

# Update layout for the stacked bar chart
fig.update_layout(
    barmode='stack',
    title='Distribusi Pelean tiap Distrik',
    xaxis_title='Distrik',
    yaxis_title='Total Pelean Diterima'
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)

#------------------------------------------------------- CLUSTERING -------------------------------------------------------------------
st.header("Pelean Clustering Using K-Means")
df_cluster = pd.melt(df, id_vars=['DISTRIK', 'RESORT', 'HURIA'], 
                     value_vars=['Total Terima Januari 2023', 'Total Terima Februari 2023',
                                'Total Terima Maret 2023', 'Total Terima April 2023',
                                'Total Terima Mei 2023', 'Total Terima Juni 2023',
                                'Total Terima Juli 2023', 'Total Terima Agustus 2023',
                                'Total Terima September 2023', 'Total Terima Oktober 2023',
                                'Total Terima November 2023', 'Total Terima Desember 2023',
                                'Total Terima Januari 2024', 'Total Terima Februari 2024',
                                'Total Terima Maret 2024', 'Total Terima April 2024',
                                'Total Terima Mei 2024', 'Total Terima Juni 2024',
                                'Total Terima Juli 2024'],
                    var_name='Kategori', value_name='Total Pelean')
df_cluster['Year'] = df_cluster['Kategori'].str[-4:]
df_cluster['Month'] = df_cluster['Kategori'].str.split(' ').str.get(2)

#------------------------------------------------ TRY ONLY CLUSTER DISTRICT --------------------------------------------------
st.subheader('Cluster Distrik')

# Slider for K selection (cluster distrik)
# st.header("K-Means Clustering Settings")
k_distrik = st.slider("Select Number of Clusters (K)", 2, 10, 5, key='slider_distrik')

# Aggregate total pelean by district
df_group_distrik = df_cluster.groupby(['DISTRIK']).agg({
        'Total Pelean': ['sum', 'mean', 'std']
    }).reset_index()

df_group_distrik.columns = ['DISTRIK', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

# Caching the clustering step
@st.cache_resource
def perform_clustering_distrik(df, k):
    # Encode 'District' as numerical values
    label_encoder = LabelEncoder()
    df['District_Encoded'] = label_encoder.fit_transform(df['DISTRIK'])

    # Prepare data for clustering
    X_distrik = df[['District_Encoded', 'Total Pelean', 'Avg Pelean', 'Std Pelean']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled_distrik = scaler.fit_transform(X_distrik)

    # Apply K-Means Clustering
    kmeans_distrik = KMeans(n_clusters=k_distrik, random_state=42)
    df['Cluster'] = kmeans_distrik.fit_predict(X_scaled_distrik)

    # Convert cluster labels to categorical for discrete coloring
    df['Cluster'] = df['Cluster'].astype(str)  # Convert to string for discrete color mapping
    return df

aggregated_data_distrik = perform_clustering_distrik(df_group_distrik, k_distrik)

# Visualize clusters
fig = px.scatter(aggregated_data_distrik, x='Total Pelean', y='Std Pelean', color='Cluster',
                #  labels={'Total Pelean': 'Total Pelean', 'Avg Pelean': 'Average Pelean'},
                 title=f"K-Means Clustering with K={k_distrik} (Only District)",
                 color_discrete_sequence=px.colors.qualitative.Plotly, # Choose a color sequence
                 labels={'Cluster': 'Cluster Label'})
st.plotly_chart(fig)

# # Count data points in each cluster
# cluster_counts_distrik = aggregated_data_distrik['Cluster'].value_counts().reset_index()
# cluster_counts_distrik.columns = ['Cluster', 'Count']

# # Display counts in Streamlit
# st.write("Count of Data Points in Each Cluster Distrik:")
# st.dataframe(cluster_counts_distrik)

#------------------------------------------------ TRY ONLY CLUSTER RESORT --------------------------------------------------
st.subheader('Cluster Resort')

# Slider for K selection (cluster resort)
# st.header("K-Means Clustering Settings")
k_resort = st.slider("Select Number of Clusters (K)", 2, 10, 3, key='slider_resort')

# Aggregate total pelean by district
df_group_resort = df_cluster.groupby(['RESORT']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

df_group_resort.columns = ['RESORT', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

# Caching the clustering step
@st.cache_resource
def perform_clustering_resort(df, k):
    # Encode 'Resort' as numerical values
    label_encoder = LabelEncoder()
    df['Resort_Encoded'] = label_encoder.fit_transform(df['RESORT'])

    # Prepare data for clustering
    X_resort = df[['Resort_Encoded', 'Total Pelean', 'Avg Pelean', 'Std Pelean']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled_resort = scaler.fit_transform(X_resort)

    # Apply K-Means Clustering
    kmeans_resort = KMeans(n_clusters=k_resort, random_state=42)
    df['Cluster'] = kmeans_resort.fit_predict(X_scaled_resort)

    # Convert cluster labels to categorical for discrete coloring
    df['Cluster'] = df['Cluster'].astype(str)  # Convert to string for discrete color mapping
    return df

aggregated_data_resort = perform_clustering_resort(df_group_resort, k_resort)

# Visualize clusters
fig = px.scatter(aggregated_data_resort, x='Total Pelean', y='Std Pelean', color='Cluster',
                #  labels={'Total Pelean': 'Total Pelean', 'Avg Pelean': 'Average Pelean'},
                 title=f"K-Means Clustering with K={k_resort} (Only Resort)",
                 color_discrete_sequence=px.colors.qualitative.Plotly, # Choose a color sequence
                 labels={'Cluster': 'Cluster Label'})
st.plotly_chart(fig)

#------------------------------------------------ TRY ONLY CLUSTER HURIA --------------------------------------------------
st.subheader('Cluster Huria')

# Slider for K selection (cluster huria)
# st.header("K-Means Clustering Settings")
k_huria = st.slider("Select Number of Clusters (K)", 2, 10, 3, key='slider_huria')

# Aggregate total pelean by district
df_group_huria = df_cluster.groupby(['HURIA']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

df_group_huria.columns = ['HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

# Caching the clustering step
@st.cache_resource
def perform_clustering_huria(df, k):
    # Encode 'Huria' as numerical values
    label_encoder = LabelEncoder()
    df['Huria_Encoded'] = label_encoder.fit_transform(df['HURIA'])

    # Prepare data for clustering
    X_huria = df[['Huria_Encoded', 'Total Pelean', 'Avg Pelean', 'Std Pelean']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled_huria = scaler.fit_transform(X_huria)

    # Apply K-Means Clustering
    kmeans_huria = KMeans(n_clusters=k_huria, random_state=42)
    df['Cluster'] = kmeans_huria.fit_predict(X_scaled_huria)

    # Convert cluster labels to categorical for discrete coloring
    df['Cluster'] = df['Cluster'].astype(str)  # Convert to string for discrete color mapping
    return df

aggregated_data_huria = perform_clustering_huria(df_group_huria, k_huria)

# Visualize clusters
fig = px.scatter(aggregated_data_huria, x='Total Pelean', y='Std Pelean', color='Cluster',
                 title=f"K-Means Clustering with K={k_huria} (Only Huria)",
                 color_discrete_sequence=px.colors.qualitative.Plotly, # Choose a color sequence
                 labels={'Cluster': 'Cluster Label'})
st.plotly_chart(fig)

#------------------------------------------------ SHOW COUNT IN EACH CLUSTER -----------------------------------------------------------
# Create columns
col1, col2, col3 = st.columns(3)

# Count data points in each cluster (distrik)
cluster_counts_distrik = aggregated_data_distrik['Cluster'].value_counts().reset_index()
cluster_counts_distrik.columns = ['Cluster', 'Count']

# Count data points in each cluster (resort)
cluster_counts_resort = aggregated_data_resort['Cluster'].value_counts().reset_index()
cluster_counts_resort.columns = ['Cluster', 'Count']

# Count data points in each cluster (huria)
cluster_counts_huria = aggregated_data_huria['Cluster'].value_counts().reset_index()
cluster_counts_huria.columns = ['Cluster', 'Count']

# Display DataFrames
with col1:
    st.write("Count of Data Points in Each Cluster Distrik:")
    st.dataframe(cluster_counts_distrik, hide_index=True)

with col2:
    st.write("Count of Data Points in Each Cluster Resort:")
    st.dataframe(cluster_counts_resort, hide_index=True)

with col3:
    st.write("Count of Data Points in Each Cluster Huria:")
    st.dataframe(cluster_counts_huria, hide_index=True)


#--------------------------------------- DETAILED BOX PLOT FOR EACH CLUSTER DISTRIK, RESORT, HURIA ---------------------------------

def create_detailed_chart(df):
    num_clusters = df['Cluster'].nunique() # jumlah cluster
    clusters = df['Cluster'].unique()
    # Create subplots: 3 plots per row
    fig = make_subplots(rows=(num_clusters // 3) + 1, cols=3, subplot_titles=[f"Cluster {i}" for i in clusters])

    # Add a boxplot for each cluster
    for i, cluster in enumerate(clusters):
        cluster_data = df[df['Cluster'] == cluster]['Total Pelean']
        row = (i // 3) + 1
        col = (i % 3) + 1

        fig.add_trace(
            go.Box(y=cluster_data, name=f'Cluster {cluster}', boxmean=True,  # Display the mean
                ),
            row=row, col=col
        )

    # Adjust layout
    fig.update_layout(height=600, width=900, title_text="Boxplots for Each Cluster (Total Pelean)", showlegend=False)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

# Expander for detailed view
with st.expander("Show Detailed Box Plot Chart"):
    fig1 = create_detailed_chart(aggregated_data_distrik)
    fig2 = create_detailed_chart(aggregated_data_resort)
    fig3 = create_detailed_chart(aggregated_data_huria)
    # Display plots
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)


#------------------------------------------------------- CLUSTER GABUNGAN -------------------------------------------------------
st.subheader('Cluster Gabungan')

# Slider for K selection (cluster gabungan)
# st.header("K-Means Clustering Settings")
k_gabung = st.slider("Select Number of Clusters (K)", 2, 10, 5, key='slider_gabung')

# Feature Engineering
# Aggregate total pelean by district, resort, and huria per year and month
aggregated_data = df_cluster.groupby(['DISTRIK', 'RESORT', 'HURIA']).agg({
    # 'Total Pelean': 'sum'
    'Total Pelean': ['sum', 'mean', 'std']
    # 'Total Pelean': ['sum', 'mean']
}).reset_index()

# Flatten MultiIndex
aggregated_data.columns = ['DISTRIK', 'RESORT', 'HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

# Encode categorical variables
encoded_data = pd.get_dummies(aggregated_data, columns=['DISTRIK', 'RESORT', 'HURIA'])

# Combine encoded categorical data with Total Pelean
X = pd.concat([encoded_data, aggregated_data[['Total Pelean', 'Avg Pelean', 'Std Pelean']]], axis=1)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=k_gabung, random_state=42)
aggregated_data['Cluster'] = kmeans.fit_predict(X_scaled)

# # Apply PCA for 2D visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# aggregated_data['PCA1'] = X_pca[:, 0]
# aggregated_data['PCA2'] = X_pca[:, 1]

# # Generate a colormap
num_clusters = aggregated_data['Cluster'].nunique() # jumlah cluster
# colors = plt.cm.get_cmap('tab10', num_clusters).colors
# color_map = {i: colors[i] for i in range(num_clusters)}

# Convert cluster labels to categorical for discrete coloring
aggregated_data['Cluster'] = aggregated_data['Cluster'].astype(str)  # Convert to string for discrete color mapping


# Plot Clusters using Plotly
fig = px.scatter(aggregated_data, x='Total Pelean', y='Std Pelean', color='Cluster',
                 hover_data=['DISTRIK', 'RESORT', 'HURIA', 'Avg Pelean'],
                 title=f"K-Means Clustering with K={k_gabung}",
                 color_discrete_sequence=px.colors.qualitative.Plotly, # Choose a color sequence
                 labels={'Cluster': 'Cluster Label'} 
                )

# # Add centroids to the plot -- byk bgt centroid nya tiap features
# for i, (x, y) in enumerate(centroids):
#     fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=12, symbol='x', color='black'),
#                              name=f'Centroid {i}'))

st.plotly_chart(fig)

# Count data points in each cluster
cluster_counts = aggregated_data['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']

# Display counts in Streamlit
st.write("Count of Data Points in Each Cluster:")
st.dataframe(cluster_counts)

# # Display Data
# st.write("Aggregated Data with Clusters", aggregated_data)

# to understand each cluster using box plots
clusters = aggregated_data['Cluster'].unique()
# Create subplots: 3 plots per row
fig = make_subplots(rows=(num_clusters // 3) + 1, cols=3, subplot_titles=[f"Cluster {i}" for i in clusters])

# Add a boxplot for each cluster
for i, cluster in enumerate(clusters):
    cluster_data = aggregated_data[aggregated_data['Cluster'] == cluster]['Total Pelean']
    row = (i // 3) + 1
    col = (i % 3) + 1

    fig.add_trace(
        go.Box(y=cluster_data, name=f'Cluster {cluster}', boxmean=True,  # Display the mean
               ),
        row=row, col=col
    )

# Adjust layout
fig.update_layout(height=600, width=900, title_text="Boxplots for Each Cluster (Total Pelean)", showlegend=False)

# Display the plot in Streamlit
st.plotly_chart(fig)

# to save the data output in csv
aggregated_data_distrik.to_csv('C:/Users/orlin/Downloads/Analisis Pelean/cluster_distrik.csv')
aggregated_data_resort.to_csv('C:/Users/orlin/Downloads/Analisis Pelean/cluster_resort.csv')
aggregated_data_huria.to_csv('C:/Users/orlin/Downloads/Analisis Pelean/cluster_huria.csv')
aggregated_data.to_csv('C:/Users/orlin/Downloads/Analisis Pelean/cluster_gabungan.csv')


#------------------------------------------------------- CLUSTER USING DBSCAN -------------------------------------------------------
# # krn pake K-means kurengggg
# # Aggregate total pelean by district, resort, and huria 
# df_DBSCAN = df_cluster.groupby(['DISTRIK', 'RESORT', 'HURIA']).agg({
#     'Total Pelean': ['sum', 'mean', 'std']
# }).reset_index()

# # Flatten MultiIndex
# df_DBSCAN.columns = ['DISTRIK', 'RESORT', 'HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

# # Apply encoding
# label_encoder = LabelEncoder()
# df_DBSCAN['DISTRIK'] = label_encoder.fit_transform(df_DBSCAN['DISTRIK'])
# df_DBSCAN['RESORT'] = label_encoder.fit_transform(df_DBSCAN['RESORT'])
# df_DBSCAN['HURIA'] = label_encoder.fit_transform(df_DBSCAN['HURIA'])

# # Features for clustering
# features = df_DBSCAN[['DISTRIK', 'RESORT', 'HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']]

# # Scale features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Apply DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=2)
# df_DBSCAN['Cluster'] = dbscan.fit_predict(scaled_features)

# # Plot 2D scatter plot
# fig = px.scatter(
#     df_DBSCAN,
#     x='DISTRIK',
#     y='RESORT',
#     color='Cluster',
#     size='Total Pelean',
#     title='DBSCAN Clustering of Data',
#     labels={'DISTRIK': 'DISTRIK', 'RESORT': 'RESORT'},
#     hover_data={'HURIA': True, 'Total Pelean': True, 'Avg Pelean': True, 'Std Pelean': True}
# )

# fig.show()

# # Count data points in each cluster
# cluster_counts_DB = df_DBSCAN['Cluster'].value_counts().reset_index()
# cluster_counts_DB.columns = ['Cluster', 'Count']

# # Display counts in Streamlit
# st.write("Count of Data Points in Each Cluster:")
# st.dataframe(cluster_counts_DB)

#---------------------------------------- EVALUATE THE NUMBER OF CLUSTERS ------------------------------------------------------------
# Aggregate total pelean by district, resort, and huria 
df_evaluate = df_cluster.groupby(['DISTRIK', 'RESORT', 'HURIA']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

# Flatten MultiIndex
df_evaluate.columns = ['DISTRIK', 'RESORT', 'HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

label_encoder = LabelEncoder()
df_evaluate['DISTRIK'] = label_encoder.fit_transform(df_evaluate['DISTRIK'])
df_evaluate['RESORT'] = label_encoder.fit_transform(df_evaluate['RESORT'])
df_evaluate['HURIA'] = label_encoder.fit_transform(df_evaluate['HURIA'])

# Features for clustering
features = df_evaluate[['DISTRIK', 'RESORT', 'HURIA', 'Total Pelean']]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

@st.cache_data
def evaluate_clusters(n_clusters_range):
    wcss = []
    silhouette_avg = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_features)
        labels = kmeans.labels_
        
        # Compute WCSS (within-cluster sum of squares)
        wcss.append(kmeans.inertia_)
        
        # Compute Silhouette Score
        if n_clusters > 1:
            silhouette_avg.append(silhouette_score(scaled_features, labels))
        else:
            silhouette_avg.append(None)
    
    return wcss, silhouette_avg

# Streamlit app
st.title('Cluster Evaluation using K-Means')
st.subheader('Evaluasi Cluster Gabungan')

# Range of clusters
n_clusters_range = st.slider('Select range of number of clusters', 2, 10, (2, 5), 1, key='slider1')

# Evaluate clustering
n_clusters_list = list(range(n_clusters_range[0], n_clusters_range[1] + 1))
wcss, silhouette_avg = evaluate_clusters(n_clusters_list)

# Convert results to DataFrames
wcss_df = pd.DataFrame({'Number of Clusters': n_clusters_list, 'WCSS': wcss})
silhouette_df = pd.DataFrame({'Number of Clusters': n_clusters_list, 'Silhouette Score': silhouette_avg})

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Score'))

# Add Elbow Method plot
fig.add_trace(
    go.Scatter(
        x=wcss_df['Number of Clusters'],
        y=wcss_df['WCSS'],
        mode='lines+markers',
        name='WCSS'
    ),
    row=1, col=1
)

# Add Silhouette Score plot
fig.add_trace(
    go.Scatter(
        x=silhouette_df['Number of Clusters'],
        y=silhouette_df['Silhouette Score'],
        mode='lines+markers',
        name='Silhouette Score'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='Cluster Evaluation',
    xaxis_title='Number of Clusters',
    yaxis_title='WCSS',
    xaxis2_title='Number of Clusters',
    yaxis2_title='Silhouette Score',
    showlegend=True
)

# Display plots
st.plotly_chart(fig, use_container_width=True)

#---------------------------------------- EVALUATE THE NUMBER OF CLUSTERS (DISTRIK) --------------------------------------------------

# Aggregate total pelean by district
df_evaluate_distrik = df_cluster.groupby(['DISTRIK']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

# Flatten MultiIndex
df_evaluate_distrik.columns = ['DISTRIK', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

label_encoder = LabelEncoder()
df_evaluate_distrik['DISTRIK'] = label_encoder.fit_transform(df_evaluate_distrik['DISTRIK'])

# Features for clustering
features_distrik = df_evaluate_distrik[['DISTRIK', 'Total Pelean']]

# Scale features
scaler = StandardScaler()
scaled_features_distrik = scaler.fit_transform(features_distrik)

@st.cache_data
def evaluate_clusters_distrik(n_clusters_range):
    wcss = []
    silhouette_avg = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_features_distrik)
        labels = kmeans.labels_
        
        # Compute WCSS (within-cluster sum of squares)
        wcss.append(kmeans.inertia_)
        
        # Compute Silhouette Score
        if n_clusters > 1:
            silhouette_avg.append(silhouette_score(scaled_features_distrik, labels))
        else:
            silhouette_avg.append(None)
    
    return wcss, silhouette_avg

# Streamlit app
st.title('Cluster Distrik Evaluation using K-Means')
st.subheader('Evaluasi Cluster Distrik')

# Range of clusters
n_clusters_range_distrik = st.slider('Select range of number of clusters', 2, 10, (2, 10), 1, key='slider2')

# Evaluate clustering
n_clusters_list_distrik = list(range(n_clusters_range_distrik[0], n_clusters_range_distrik[1] + 1))
wcss, silhouette_avg = evaluate_clusters_distrik(n_clusters_list_distrik)

# Convert results to DataFrames
wcss_df = pd.DataFrame({'Number of Clusters': n_clusters_list_distrik, 'WCSS': wcss})
silhouette_df = pd.DataFrame({'Number of Clusters': n_clusters_list_distrik, 'Silhouette Score': silhouette_avg})

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Score'))

# Add Elbow Method plot
fig.add_trace(
    go.Scatter(
        x=wcss_df['Number of Clusters'],
        y=wcss_df['WCSS'],
        mode='lines+markers',
        name='WCSS'
    ),
    row=1, col=1
)

# Add Silhouette Score plot
fig.add_trace(
    go.Scatter(
        x=silhouette_df['Number of Clusters'],
        y=silhouette_df['Silhouette Score'],
        mode='lines+markers',
        name='Silhouette Score'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='Cluster Distrik Evaluation',
    xaxis_title='Number of Clusters',
    yaxis_title='WCSS',
    xaxis2_title='Number of Clusters',
    yaxis2_title='Silhouette Score',
    showlegend=True
)

# Display plots
st.plotly_chart(fig, use_container_width=True)

#---------------------------------------- EVALUATE THE NUMBER OF CLUSTERS (RESORT) --------------------------------------------------

# Aggregate total pelean by district
df_evaluate_resort = df_cluster.groupby(['RESORT']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

# Flatten MultiIndex
df_evaluate_resort.columns = ['RESORT', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

label_encoder = LabelEncoder()
df_evaluate_resort['RESORT'] = label_encoder.fit_transform(df_evaluate_resort['RESORT'])

# Features for clustering
features_resort = df_evaluate_resort[['RESORT', 'Total Pelean']]

# Scale features
scaler = StandardScaler()
scaled_features_resort = scaler.fit_transform(features_resort)

@st.cache_data
def evaluate_clusters_resort(n_clusters_range):
    wcss = []
    silhouette_avg = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_features_resort)
        labels = kmeans.labels_
        
        # Compute WCSS (within-cluster sum of squares)
        wcss.append(kmeans.inertia_)
        
        # Compute Silhouette Score
        if n_clusters > 1:
            silhouette_avg.append(silhouette_score(scaled_features_resort, labels))
        else:
            silhouette_avg.append(None)
    
    return wcss, silhouette_avg

# Streamlit app
st.title('Cluster Resort Evaluation using K-Means')
st.subheader('Evaluasi Cluster Resort')

# Range of clusters
n_clusters_range_resort = st.slider('Select range of number of clusters', 2, 10, (2, 10), 1, key='slider3')

# Evaluate clustering
n_clusters_list_resort = list(range(n_clusters_range_resort[0], n_clusters_range_resort[1] + 1))
wcss, silhouette_avg = evaluate_clusters_resort(n_clusters_list_resort)

# Convert results to DataFrames
wcss_df = pd.DataFrame({'Number of Clusters': n_clusters_list_resort, 'WCSS': wcss})
silhouette_df = pd.DataFrame({'Number of Clusters': n_clusters_list_resort, 'Silhouette Score': silhouette_avg})

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Score'))

# Add Elbow Method plot
fig.add_trace(
    go.Scatter(
        x=wcss_df['Number of Clusters'],
        y=wcss_df['WCSS'],
        mode='lines+markers',
        name='WCSS'
    ),
    row=1, col=1
)

# Add Silhouette Score plot
fig.add_trace(
    go.Scatter(
        x=silhouette_df['Number of Clusters'],
        y=silhouette_df['Silhouette Score'],
        mode='lines+markers',
        name='Silhouette Score'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='Cluster Resort Evaluation',
    xaxis_title='Number of Clusters',
    yaxis_title='WCSS',
    xaxis2_title='Number of Clusters',
    yaxis2_title='Silhouette Score',
    showlegend=True
)

# Display plots
st.plotly_chart(fig, use_container_width=True)

#---------------------------------------- EVALUATE THE NUMBER OF CLUSTERS (HURIA) --------------------------------------------------

# Aggregate total pelean by district
df_evaluate_huria = df_cluster.groupby(['HURIA']).agg({
    'Total Pelean': ['sum', 'mean', 'std']
}).reset_index()

# Flatten MultiIndex
df_evaluate_huria.columns = ['HURIA', 'Total Pelean', 'Avg Pelean', 'Std Pelean']

label_encoder = LabelEncoder()
df_evaluate_huria['HURIA'] = label_encoder.fit_transform(df_evaluate_huria['HURIA'])

# Features for clustering
features_huria = df_evaluate_huria[['HURIA', 'Total Pelean']]

# Scale features
scaler = StandardScaler()
scaled_features_huria = scaler.fit_transform(features_huria)

@st.cache_data
def evaluate_clusters_huria(n_clusters_range):
    wcss = []
    silhouette_avg = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_features_huria)
        labels = kmeans.labels_
        
        # Compute WCSS (within-cluster sum of squares)
        wcss.append(kmeans.inertia_)
        
        # Compute Silhouette Score
        if n_clusters > 1:
            silhouette_avg.append(silhouette_score(scaled_features_huria, labels))
        else:
            silhouette_avg.append(None)
    
    return wcss, silhouette_avg

# Streamlit app
st.title('Cluster Huria Evaluation using K-Means')
st.subheader('Evaluasi Cluster Huria')

# Range of clusters
n_clusters_range_huria = st.slider('Select range of number of clusters', 2, 10, (2, 10), 1, key='slider4')

# Evaluate clustering
n_clusters_list_huria = list(range(n_clusters_range_huria[0], n_clusters_range_huria[1] + 1))
wcss, silhouette_avg = evaluate_clusters_huria(n_clusters_list_huria)

# Convert results to DataFrames
wcss_df = pd.DataFrame({'Number of Clusters': n_clusters_list_huria, 'WCSS': wcss})
silhouette_df = pd.DataFrame({'Number of Clusters': n_clusters_list_huria, 'Silhouette Score': silhouette_avg})

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Score'))

# Add Elbow Method plot
fig.add_trace(
    go.Scatter(
        x=wcss_df['Number of Clusters'],
        y=wcss_df['WCSS'],
        mode='lines+markers',
        name='WCSS'
    ),
    row=1, col=1
)

# Add Silhouette Score plot
fig.add_trace(
    go.Scatter(
        x=silhouette_df['Number of Clusters'],
        y=silhouette_df['Silhouette Score'],
        mode='lines+markers',
        name='Silhouette Score'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='Cluster Huria Evaluation',
    xaxis_title='Number of Clusters',
    yaxis_title='WCSS',
    xaxis2_title='Number of Clusters',
    yaxis2_title='Silhouette Score',
    showlegend=True
)

# Display plots
st.plotly_chart(fig, use_container_width=True)


  

