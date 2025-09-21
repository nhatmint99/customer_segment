# In[]

import pandas as pd

# Load datasets
transactions = pd.read_csv("Transactions.csv")
products = pd.read_csv("Products_with_Categories.csv")

print("Transactions shape:", transactions.shape)
print("Products shape:", products.shape)

transactions.head(), products.head()

# In[]

import numpy as np

# Merge datasets
df = transactions.merge(products, on="productId", how="left")

# RFM calculation
df['TransactionDate'] = pd.to_datetime(df['Date'])
max_date = df['TransactionDate'].max()
df['Amount'] = df['items'] * df['price']
customer_features = df.groupby("Member_number").agg({
    "TransactionDate": lambda x: (max_date - x.max()).days,  # Recency
    "productId": "count",                               # Frequency
    "Amount": "sum",                                        # Monetary
    "Category": pd.Series.nunique                           # Category Diversity
}).reset_index()

customer_features.columns = ["Member_number", "Recency", "Frequency", "Monetary", "Category_Diversity"]
customer_features.head()

# In[]
import matplotlib.pyplot as plt

# In[]
# Vẽ phân phối của 'Recency'
plt.subplot(3, 1, 1) # 3 hàng, 1 cột, vị trí thứ nhất
plt.hist(customer_features['Recency'], bins=20, edgecolor='black') # Chọn số lượng bins phù hợp
plt.title('Distribution of Recency')
plt.xlabel('Recency')

# Vẽ phân phối của 'Frequency'
plt.subplot(3, 1, 2) # 3 hàng, 1 cột, vị trí thứ hai
plt.hist(customer_features['Frequency'], bins=20, edgecolor='black') # Chọn số lượng bins phù hợp
plt.title('Distribution of Frequency')
plt.xlabel('Frequency')

# Vẽ phân phối của 'Monetary'
plt.subplot(3, 1, 3) # 3 hàng, 1 cột, vị trí thứ ba
plt.hist(customer_features['Monetary'], bins=20, edgecolor='black') # Chọn số lượng bins phù hợp
plt.title('Distribution of Monetary')
plt.xlabel('Monetary')

plt.tight_layout()
plt.show()

# In[]
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

# In[]
scaler = StandardScaler()
X = scaler.fit_transform(customer_features.drop("Member_number", axis=1))


# In[]
# KMeans
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_features['cluster_kmeans'] = kmeans_model.fit_predict(X)

# GMM
gmm_model = GaussianMixture(n_components=4, random_state=42)
customer_features['cluster_gmm'] = gmm_model.fit_predict(X)

# Agglomerative
agg_model = AgglomerativeClustering(n_clusters=4)
customer_features['cluster_agg'] = agg_model.fit_predict(X)

# Hierarchical (Scipy)
Z = linkage(X, method='ward')
customer_features['cluster_hier'] = fcluster(Z, 4, criterion='maxclust')

# In[]
# Plot dendrogram (all customers - may be very dense)
plt.figure(figsize=(12,6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram (truncated)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()



# In[]
# Manual RFM segmentation into 4 clusters
customer_features["R_quartile"] = pd.qcut(customer_features["Recency"], 4, labels=[4,3,2,1])
customer_features["F_quartile"] = pd.qcut(customer_features["Frequency"], 4, labels=[1,2,3,4])
customer_features["M_quartile"] = pd.qcut(customer_features["Monetary"], 4, labels=[1,2,3,4])

customer_features["RFM_Score"] = customer_features[["R_quartile","F_quartile","M_quartile"]].astype(int).sum(axis=1)
customer_features["cluster_rfm"] = pd.qcut(customer_features["RFM_Score"], 4, labels=[0,1,2,3])

customer_features.head()

# In[]
def segment_customer(rfm_score):
    if rfm_score >= 10:
        return 'VIP'
    elif rfm_score >= 7:
        return 'Loyal Customers'
    elif rfm_score >= 4:
        return 'Ordinary Buyers'
    else:
        return 'Lost Customers'
customer_features1 = customer_features.copy()
customer_features1['Segment'] = customer_features1['RFM_Score'].apply(segment_customer)
customer_features1.head(10)


# In[]
customer_features1["Percent"] = (customer_features1.groupby("Segment")["Member_number"].transform("count") / len(customer_features1)) * 100
customer_features1.head(10)

# In[]
customer_features1 = customer_features1.groupby("Segment").agg({
    "Member_number": "count",
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Category_Diversity": "mean",
    "Percent": "unique"})
customer_features1.head(10)

# In[]


# In[]
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.clustering import KMeans as SparkKMeans

# In[]
# Start Spark
spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()


# In[]
# Convert Pandas -> Spark
spark_df = spark.createDataFrame(customer_features.drop(columns=[
    "cluster_kmeans","cluster_gmm","cluster_agg","cluster_hier","cluster_rfm",
    "R_quartile","F_quartile","M_quartile","RFM_Score"
]))

# In[]
# Vector assembler
assembler = VectorAssembler(inputCols=["Recency","Frequency","Monetary","Category_Diversity"], outputCol="features")
assembled = assembler.transform(spark_df).cache()

scaler_spark = SparkScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model_spark = scaler_spark.fit(assembled)
scaled = scaler_model_spark.transform(assembled).cache()

# Spark KMeans
spark_kmeans = SparkKMeans(k=4, seed=42, featuresCol="scaled_features", predictionCol="cluster_spark")
spark_kmeans_model = spark_kmeans.fit(scaled)
spark_clustered = spark_kmeans_model.transform(scaled)

# Convert back to Pandas
df_spark_result = spark_clustered.select("Member_number", "cluster_spark").toPandas()

# Merge results
customer_features = customer_features.merge(df_spark_result, on="Member_number")
customer_features.head()

# In[]
method = customer_features[['Member_number','cluster_kmeans', 'cluster_gmm', 'cluster_agg', 'cluster_hier', 'cluster_rfm', 'cluster_spark','RFM_Score']]
method = method.rename(columns={
    'Member_number': 'Member',
    'cluster_kmeans': 'KMeans',
    'cluster_gmm': 'GMM',
    'cluster_agg': 'Agglomerative',
    'cluster_hier': 'Hierarchical',
    'cluster_rfm': 'RFM',
    'cluster_spark': 'Spark KMeans'
})
method_segment = method['RFM_Score'].apply(segment_customer)
method['RFM_Segment'] = method_segment
method

# In[]
# Optimized percentage calculation for all methods
def calculate_method_percentages(method_df, method_columns):
    """
    Efficiently calculate percentages for all clustering methods
    """
    for method_col in method_columns:
        percent_col = f"{method_col}_Percent"
        method_df[percent_col] = (
            method_df.groupby(method_col)[method_col].transform("count") / len(method_df) * 100
        ).round(2)
    return method_df

# Define method columns
method_columns = ["KMeans", "GMM", "Agglomerative", "Hierarchical", "RFM", "SparkKMeans"]

# Calculate percentages efficiently
method = calculate_method_percentages(method, method_columns)

print("✅ Percentages calculated for all methods")
method.head(10)

# In[]
method.to_csv("Clustering_Method_Comparison.csv", index=False)

# In[]
method['Spark KMeans'].unique()

# In[]
method = method.rename(columns={'Spark KMeans': 'SparkKMeans'})
method

# In[]
# Optimized method summary generation
def create_method_summaries(method_df, method_names):
    """
    Create summaries for all clustering methods efficiently
    """
    summaries = {}
    
    for method_name in method_names:
        percent_col = "SparkKMeans_Percent" if method_name == "SparkKMeans" else f"{method_name}_Percent"
        
        summary = method_df.groupby(method_name).agg({
            "Member": "count",
            percent_col: "mean"
        }).reset_index()
        
        summary = summary.rename(columns={method_name: "Cluster"})
        summary["Cluster"] = summary["Cluster"].astype(str)
        summary[percent_col] = summary[percent_col].round(2)
        
        summaries[method_name] = summary
    
    return summaries

# Create summaries for all methods
method_names = ["KMeans", "GMM", "Agglomerative", "Hierarchical", "RFM", "SparkKMeans"]
method_summaries = create_method_summaries(method, method_names)

# Create consolidated comparison dataframe
method_comparison = pd.DataFrame()
method_comparison["Cluster"] = [0, 1, 2, 3]

for method_name in method_names:
    percent_col = "SparkKMeans_Percent" if method_name == "SparkKMeans" else f"{method_name}_Percent"
    method_comparison[method_name] = method_summaries[method_name][percent_col].values

print("📊 Method comparison table created:")
method_comparison

# In[]
method_sum = pd.DataFrame()
method_sum["Cluster"] = [0,1,2,3]
method_sum["RFM_Manual"] = method_sum_rfm["RFM_Percent"]
method_sum["KMeans"] = method_sum_kmeans["KMeans_Percent"]
method_sum["GMM"] = method_sum_gmm["GMM_Percent"]
method_sum["Agglomerative"] = method_sum_agg["Agglomerative_Percent"]
method_sum["Hierarchical"] = method_sum_hier["Hierarchical_Percent"]
method_sum["SparkKMeans"] = method_sum_spark["SparkKMeans_Percent"]
method_sum

# In[]
import squarify

# In[]
list_cus = customer_features1.index.tolist()
list_cus

# In[]
# Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(14, 10)

colors_dict = {'Lost Customers':'green','Loyal Customers':'royalblue',
               'Ordinary Buyers':'red','VIP':'gold'}

sizes = customer_features1["Member_number"].values
labels = [
    f"{seg}\n{row['Recency']:.0f} days\n{row['Frequency']:.0f} orders\n{row['Monetary']:.0f} $\n{row['Member_number']} customers ({row['Percent'][0]:.2f}%)"
    for seg, row in customer_features1.iterrows()
]

squarify.plot(
    sizes=sizes,
    text_kwargs={'fontsize':10,'weight':'bold', 'fontname':"sans serif"},
    color=[colors_dict[seg] for seg in customer_features1.index],
    label=labels,
    alpha=0.5
)

plt.title("Customers Segments", fontsize=22, fontweight="bold")
plt.axis('off')

plt.savefig('RFM_Segments.png')
plt.show()

# In[]
# Optimized cluster profile creation for all methods
def create_all_cluster_profiles():
    """
    Create cluster profiles for all methods efficiently using vectorized operations
    """
    method_configs = {
        "KMeans": "cluster_kmeans",
        "GMM": "cluster_gmm", 
        "Agglomerative": "cluster_agg",
        "Hierarchical": "cluster_hier",
        "SparkKMeans": "cluster_spark"
    }
    
    profiles = {}
    total_customers = len(customer_features)
    
    for method_name, cluster_column in method_configs.items():
        # Vectorized aggregation
        profile = customer_features.groupby(cluster_column).agg({
            "Member_number": "count",
            "Recency": "mean", 
            "Frequency": "mean",
            "Monetary": "mean"
        }).reset_index()
        
        # Vectorized percentage calculation
        profile["Percent"] = (profile["Member_number"] / total_customers * 100).round(2)
        
        # Rename cluster column for consistency
        profile = profile.rename(columns={cluster_column: "Cluster"})
        profile["Method"] = method_name
        
        profiles[method_name] = profile
    
    return profiles

# Create all profiles efficiently
all_profiles = create_all_cluster_profiles()

print("✅ All cluster profiles created efficiently:")
for method_name, profile in all_profiles.items():
    print(f"📈 {method_name}: {len(profile)} clusters")

# Extract individual profiles for compatibility
kmeans_profile = all_profiles["KMeans"]
gmm_profile = all_profiles["GMM"]
agg_profile = all_profiles["Agglomerative"] 
hier_profile = all_profiles["Hierarchical"]
spark_profile = all_profiles["SparkKMeans"]

# In[]
# Optimized treemap visualization function with better performance
def create_optimized_treemap(profile_data, method_name, save_name, show_plot=True):
    """
    Create optimized treemap visualization with improved performance and customization
    """
    # Clear previous plots to save memory
    plt.clf()
    
    # Create figure with optimal size
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    
    # Optimized color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Vectorized data preparation
    sizes = profile_data["Member_number"].values
    cluster_colors = colors[:len(profile_data)]
    
    # Optimized label creation using list comprehension
    labels = [
        f"Cluster {row['Cluster']}\n"
        f"{row['Recency']:.0f} days\n"
        f"{row['Frequency']:.0f} orders\n"
        f"${row['Monetary']:.0f}\n"
        f"{row['Member_number']} customers ({row['Percent']:.1f}%)"
        for _, row in profile_data.iterrows()
    ]
    
    # Create treemap with optimized parameters
    squarify.plot(
        sizes=sizes,
        text_kwargs={'fontsize': 9, 'weight': 'bold', 'fontfamily': 'sans-serif'},
        color=cluster_colors,
        label=labels,
        alpha=0.75,
        ax=ax
    )
    
    # Optimize title and layout
    ax.set_title(f"Customer Segments - {method_name}", 
                fontsize=20, fontweight="bold", pad=20)
    ax.axis('off')
    
    # Save with optimized settings
    if save_name:
        plt.savefig(f'{save_name}_Segments.png', 
                   dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Free memory if not showing
    
    return fig

# Batch create all treemaps efficiently
def create_all_treemaps(profiles_dict, show_individual=True):
    """
    Create all treemap visualizations efficiently
    """
    method_configs = {
        "KMeans": "KMeans Clustering",
        "GMM": "Gaussian Mixture Model",
        "Agglomerative": "Agglomerative Clustering", 
        "Hierarchical": "Hierarchical Clustering",
        "SparkKMeans": "Spark KMeans Clustering"
    }
    
    figures = {}
    
    for method_key, display_name in method_configs.items():
        if method_key in profiles_dict:
            fig = create_optimized_treemap(
                profiles_dict[method_key], 
                display_name, 
                method_key,
                show_plot=show_individual
            )
            figures[method_key] = fig
    
    return figures

print("🚀 Optimized treemap functions created")
print("📊 Ready to generate all visualizations efficiently")

# In[]
# Generate all treemaps efficiently in batch
print("🎨 Generating all treemap visualizations...")

# Create all treemaps at once
treemap_figures = create_all_treemaps(all_profiles, show_individual=True)

print(f"✅ Generated {len(treemap_figures)} treemap visualizations successfully!")
print("📁 PNG files saved for each method")

# In[]
# Optimized comparison grid visualization
def create_optimized_comparison_grid(profiles_dict):
    """
    Create optimized comparison grid with better layout and performance
    """
    # Clear any existing plots
    plt.clf()
    
    # Create optimized subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    fig.suptitle("Customer Segmentation Comparison - All Methods", 
                fontsize=22, fontweight="bold", y=0.95)
    
    # Define method order and layout
    method_positions = [
        ("KMeans", axes[0,0]),
        ("GMM", axes[0,1]), 
        ("Agglomerative", axes[0,2]),
        ("Hierarchical", axes[1,0]),
        ("SparkKMeans", axes[1,1])
    ]
    
    # Optimized color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Create each subplot efficiently
    for method_name, ax in method_positions:
        if method_name in profiles_dict:
            profile_data = profiles_dict[method_name]
            
            # Vectorized data preparation
            sizes = profile_data["Member_number"].values
            subplot_colors = colors[:len(profile_data)]
            
            # Compact labels for grid view
            labels = [
                f"C{row['Cluster']}\n{row['Member_number']}\n({row['Percent']:.1f}%)"
                for _, row in profile_data.iterrows()
            ]
            
            # Create treemap subplot
            squarify.plot(
                sizes=sizes,
                text_kwargs={'fontsize': 7, 'weight': 'bold'},
                color=subplot_colors,
                label=labels,
                alpha=0.8,
                ax=ax
            )
            
            ax.set_title(method_name, fontsize=12, fontweight="bold", pad=10)
            ax.axis('off')
    
    # Hide unused subplot
    axes[1,2].axis('off')
    
    # Optimize layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save with optimized settings
    plt.savefig('All_Methods_Comparison_Optimized.png', 
               dpi=200, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    plt.show()
    return fig

# Create optimized comparison grid
print("📊 Creating optimized comparison grid...")
comparison_grid_fig = create_optimized_comparison_grid(all_profiles)
print("✅ Comparison grid created and saved!")

# In[]


# In[]

import joblib

# Save sklearn models
joblib.dump(kmeans_model, "kmeans_model.pkl")
joblib.dump(gmm_model, "gmm_model.pkl")
joblib.dump(agg_model, "agg_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# In[]

# Save Spark model (overwrite if exists)
spark_kmeans_model.write().overwrite().save("spark_kmeans_model")

# In[]
import plotly.express as px

# In[]
# Create RFM averages data for visualization
rfm_avg = customer_features1.reset_index()
rfm_avg = rfm_avg.rename(columns={
    'Recency': 'RecencyMean',
    'Frequency': 'FrequencyMean', 
    'Monetary': 'MonetaryMean',
    'index': 'Segment'
})
        
# Enhanced interactive scatter plot function
def create_enhanced_scatter_plot(rfm_data, title_suffix=""):
    """
    Create enhanced interactive scatter plot for RFM analysis
    """
    fig = px.scatter(
        rfm_data, 
        x="RecencyMean", 
        y="MonetaryMean", 
        size="FrequencyMean", 
        color="Segment",
        hover_name="Segment",
        hover_data={
            'RecencyMean': ':.1f',
            'MonetaryMean': ':.0f', 
            'FrequencyMean': ':.1f'
        },
        size_max=60,
        title=f"RFM Customer Segments Analysis{title_suffix}",
        labels={
            'RecencyMean': 'Average Recency (Days)',
            'MonetaryMean': 'Average Monetary Value ($)',
            'FrequencyMean': 'Average Frequency'
        },
        color_discrete_map={
            'VIP': '#FFD700',
            'Loyal Customers': '#4169E1', 
            'Ordinary Buyers': '#FF4500',
            'Lost Customers': '#32CD32'
        }
    )
    
    fig.update_layout(
        width=900,
        height=600,
        showlegend=True,
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig

# Create and display the plot
fig = create_enhanced_scatter_plot(rfm_avg)
fig.show()

# In[]
# Optimized function to create scatter plots for all clustering methods
def create_method_comparison_plots():
    """
    Create scatter plots comparing all clustering methods
    """
    # Define method mappings
    method_mappings = {
        'KMeans': 'cluster_kmeans',
        'GMM': 'cluster_gmm', 
        'Agglomerative': 'cluster_agg',
        'Hierarchical': 'cluster_hier',
        'Spark KMeans': 'cluster_spark'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(method_mappings.keys()),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    row_col_pairs = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for idx, (method_name, cluster_col) in enumerate(method_mappings.items()):
        if idx < len(row_col_pairs):
            row, col = row_col_pairs[idx]
            
            # Create profile for this method
            method_profile = customer_features.groupby(cluster_col).agg({
                "Recency": "mean",
                "Frequency": "mean", 
                "Monetary": "mean",
                "Member_number": "count"
            }).reset_index()
            
            # Add scatter plot
            for cluster_idx, cluster_row in method_profile.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[cluster_row['Recency']],
                        y=[cluster_row['Monetary']],
                        mode='markers',
                        marker=dict(
                            size=cluster_row['Frequency']*3,
                            color=colors[cluster_idx % len(colors)],
                            opacity=0.7
                        ),
                        name=f"Cluster {cluster_row[cluster_col]}",
                        showlegend=False
                    ),
                    row=row, col=col
                )
    fig.update_layout(
        height=800,
        title_text="Customer Segmentation Comparison - All Methods",
        title_font_size=16,
        showlegend=True
    )

    
    return fig


# Import required plotly modules
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create and display comparison plots
comparison_fig = create_method_comparison_plots()
comparison_fig.show()

# In[]

def predict_new_customer(features, method="kmeans"):
    values = np.array([[features["Recency"], features["Frequency"], features["Monetary"], features["Category_Diversity"]]])
    
    if method == "kmeans":
        X_scaled = scaler.transform(values)
        return kmeans_model.predict(X_scaled)[0]
    elif method == "gmm":
        X_scaled = scaler.transform(values)
        return gmm_model.predict(X_scaled)[0]
    elif method == "agg":
        X_scaled = scaler.transform(values)
        return agg_model.fit_predict(X_scaled)[0]
    elif method == "hier":
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(X, method='ward')
        return fcluster(Z, 4, criterion='maxclust')[0]
    elif method == "rfm":
        return int(pd.qcut([features["Recency"]+features["Frequency"]+features["Monetary"]], 4, labels=[0,1,2,3])[0])
    elif method == "spark":
        new_df = spark.createDataFrame([features])
        assembled = assembler.transform(new_df)
        scaled = scaler_model_spark.transform(assembled)
        prediction = spark_kmeans_model.transform(scaled).collect()[0]["cluster_spark"]
        return prediction


# In[]

# Example
new_customer = {"Recency": 30, "Frequency": 5, "Monetary": 1200, "Category_Diversity": 3}
print("Predicted cluster (KMeans):", predict_new_customer(new_customer, "kmeans"))
print("Predicted cluster (Spark):", predict_new_customer(new_customer, "spark"))

# In[]
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


# In[]




# In[]
# Optimized clustering evaluation and comparison
def evaluate_clustering_performance(scaled_features, customer_data):
    """
    Efficiently evaluate all clustering methods with comprehensive metrics
    """
    # Define clustering methods and their columns
    clustering_methods = {
        "KMeans": "cluster_kmeans",
        "GMM": "cluster_gmm",
        "Agglomerative": "cluster_agg", 
        "Hierarchical": "cluster_hier",
        "SparkKMeans": "cluster_spark"
    }
    
    # Calculate silhouette scores efficiently
    silhouette_scores = {}
    for method_name, cluster_col in clustering_methods.items():
        if cluster_col in customer_data.columns:
            score = silhouette_score(scaled_features, customer_data[cluster_col])
            silhouette_scores[method_name] = round(score, 4)
    
    # Calculate pairwise similarity metrics efficiently
    method_columns = list(clustering_methods.values())
    similarity_results = []
    
    for i in range(len(method_columns)):
        for j in range(i+1, len(method_columns)):
            col1, col2 = method_columns[i], method_columns[j]
            
            if col1 in customer_data.columns and col2 in customer_data.columns:
                # Vectorized metric calculations
                ari = adjusted_rand_score(customer_data[col1], customer_data[col2])
                nmi = normalized_mutual_info_score(customer_data[col1], customer_data[col2])
                
                similarity_results.append([
                    col1.replace('cluster_', '').title(),
                    col2.replace('cluster_', '').title(), 
                    round(ari, 4),
                    round(nmi, 4)
                ])
    
    # Create results dataframe
    similarity_df = pd.DataFrame(
        similarity_results, 
        columns=["Method A", "Method B", "ARI", "NMI"]
    )
    
    return silhouette_scores, similarity_df

# Perform optimized evaluation
print("🔍 Evaluating clustering performance...")
optimized_silhouette_scores, optimized_similarity_df = evaluate_clustering_performance(X, customer_features)
print("📊 Silhouette Scores:")
for method, score in optimized_silhouette_scores.items():
    print(f"   {method}: {score:.4f}")



# In[]
print(f"\n📈 Similarity Analysis ({len(optimized_similarity_df)} comparisons):")
print(optimized_similarity_df.round(3))

# In[]
optimized_similarity_df

# In[]
profile_methods = {
    "KMeans": "cluster_kmeans",
    "GMM": "cluster_gmm",
    "Agglomerative": "cluster_agg",
    "Hierarchical": "cluster_hier",
    "RFM": "cluster_rfm",
    "SparkKMeans": "cluster_spark"
}

profiles = []
for method, col in profile_methods.items():
    means = customer_features.groupby(col)[["Recency","Frequency","Monetary","Category_Diversity"]].mean().reset_index()
    means["Method"] = method
    profiles.append(means)

profiles_df = pd.concat(profiles)
profiles_df

# In[]

df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with KMeans clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_kmeans"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_kmeans"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_kmeans", marker="o")
plt.title("Monthly Spend Trend by Cluster (KMeans Example)")
plt.xticks(rotation=45)
plt.show()

# In[]
customer_features.columns

# In[]

df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with GMM clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_gmm"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_gmm"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_gmm", marker="o")
plt.title("Monthly Spend Trend by Cluster (GMM Example)")
plt.xticks(rotation=45)
plt.show()

# In[]

df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with Agg clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_agg"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_agg"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_agg", marker="o")
plt.title("Monthly Spend Trend by Cluster (Agg Example)")
plt.xticks(rotation=45)
plt.show()

# In[]

df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with Hierarchical clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_hier"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_hier"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_hier", marker="o")
plt.title("Monthly Spend Trend by Cluster (Hierarchical Example)")
plt.xticks(rotation=45)
plt.show()

# In[]


df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with RFM clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_rfm"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_rfm"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_rfm", marker="o")
plt.title("Monthly Spend Trend by Cluster (RFM Example)")
plt.xticks(rotation=45)
plt.show()

# In[]

df['YearMonth'] = df['TransactionDate'].dt.to_period('M').astype(str)

# Example with Spark KMeans clusters (can replicate for others)
df_trend = df.merge(customer_features[["Member_number","cluster_spark"]], on="Member_number")
trend = df_trend.groupby(["YearMonth","cluster_spark"])["Amount"].sum().reset_index()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x="YearMonth", y="Amount", hue="cluster_spark", marker="o")
plt.title("Monthly Spend Trend by Cluster (Spark KMeans Example)")
plt.xticks(rotation=45)
plt.show()

# In[]

customer_features.to_csv("customer_segmentation_results.csv", index=False)
similarity_df.to_csv("clustering_comparison_scores.csv", index=False)
profiles_df.to_csv("clustering_profiles.csv", index=False)

print("Results exported: segmentation_results.csv, comparison_scores.csv, profiles.csv")

# In[]

# Performance summary
performance = pd.DataFrame([
    ["KMeans", silhouette_scores["KMeans"], 4],
    ["GMM", silhouette_scores["GMM"], 4],
    ["Agglomerative", silhouette_scores["Agglomerative"], 4],
    ["Hierarchical", silhouette_scores["Hierarchical"], 4],
    ["SparkKMeans", silhouette_scores["SparkKMeans"], 4],
    ["RFM", None, 4]
], columns=["Method","Silhouette","Clusters"])

print("Performance Summary:")
print(performance)

best_method = performance.loc[performance["Silhouette"].idxmax()]
print(f"\n✅ Best Method: {best_method['Method']}")
print(f"Reason: Highest silhouette score ({best_method['Silhouette']:.2f}) with {best_method['Clusters']} clusters.")
print("Alternative: RFM for interpretability, SparkKMeans for scalability.")

# In[]


