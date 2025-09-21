import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
# from pyspark.ml.clustering import KMeans as SparkKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df_products = pd.read_csv("Products_with_Categories.csv")
    df_trans = pd.read_csv("Transactions.csv")
    return df_products, df_trans

df_products, df_trans = load_data()

def customer_features(products, transactions):
    df = transactions.merge(products, on="productId", how="left")
    # RFM calculation
    df['TransactionDate'] = pd.to_datetime(df['Date'])
    max_date = df['TransactionDate'].max()
    df['Amount'] = df['items'] * df['price']
    customer_features = df.groupby("Member_number").agg({
        "TransactionDate": lambda x: (max_date - x.max()).days,  # Recency
        "productId": "count",                                    # Frequency
        "Amount": "sum",                                         # Monetary
        "Category": pd.Series.nunique                            # Category Diversity
    }).reset_index()

    customer_features.columns = ["Member_number", "Recency", "Frequency", "Monetary", "Category_Diversity"]
        # ✅ Fill NaN trước khi scale
    customer_features = customer_features.fillna({
        "Recency": customer_features["Recency"].max(),  # giả sử khách chưa mua → recency lớn nhất
        "Frequency": 0,
        "Monetary": 0,
        "Category_Diversity": 0
    })
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(customer_features.drop("Member_number", axis=1))

    # --- Clustering ---
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

    # Manual RFM segmentation
    #  --- Manual RFM segmentation ---
    try:
        r_labels = pd.qcut(customer_features["Recency"], 4, labels=[4,3,2,1], duplicates='drop')
        f_labels = pd.qcut(customer_features["Frequency"], 4, labels=[1,2,3,4], duplicates='drop')
        m_labels = pd.qcut(customer_features["Monetary"], 4, labels=[1,2,3,4], duplicates='drop')

        customer_features["R_quartile"] = r_labels.cat.codes.replace(-1, 0)  # -1 -> 0
        customer_features["F_quartile"] = f_labels.cat.codes.replace(-1, 0)
        customer_features["M_quartile"] = m_labels.cat.codes.replace(-1, 0)

        customer_features["RFM_Score"] = (
            customer_features["R_quartile"] + 
            customer_features["F_quartile"] + 
            customer_features["M_quartile"]
        )
        customer_features["cluster_rfm"] = pd.qcut(customer_features["RFM_Score"], 4, labels=[0,1,2,3], duplicates='drop')
    except Exception as e:
        print(f"⚠️ Lỗi khi tính RFM quartiles: {e}")
        customer_features["R_quartile"] = 0
        customer_features["F_quartile"] = 0
        customer_features["M_quartile"] = 0
        customer_features["RFM_Score"] = 0
        customer_features["cluster_rfm"] = 0
    # Spark ML
    # spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()
    # spark_df = spark.createDataFrame(customer_features.drop(columns=[
    #     "cluster_kmeans","cluster_gmm","cluster_agg","cluster_hier","cluster_rfm",
    #     "R_quartile","F_quartile","M_quartile","RFM_Score"
    # ]))

    # assembler = VectorAssembler(inputCols=["Recency","Frequency","Monetary","Category_Diversity"], outputCol="features")
    # assembled = assembler.transform(spark_df).cache()

    # scaler_spark = SparkScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    # scaler_model_spark = scaler_spark.fit(assembled)
    # scaled = scaler_model_spark.transform(assembled).cache()

    # spark_kmeans = SparkKMeans(k=4, seed=42, featuresCol="scaled_features", predictionCol="cluster_spark")
    # spark_kmeans_model = spark_kmeans.fit(scaled)
    # spark_clustered = spark_kmeans_model.transform(scaled)

    # df_spark_result = spark_clustered.select("Member_number", "cluster_spark").toPandas()
    # customer_features = customer_features.merge(df_spark_result, on="Member_number")

    return customer_features, X
# ===============================
# Compute RFM scores automatically
# ===============================
def compute_rfm_scores(df, user_id_col="Member_number", date_col="Date", prod_col="productId", amount_col=None):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Frequency = number of purchases
    freq = df.groupby(user_id_col)[prod_col].count()

    # Monetary
    if amount_col and amount_col in df.columns:
        monetary = df.groupby(user_id_col)[amount_col].sum()
    else:
        monetary = freq.copy()

    # Recency
    rec = df.groupby(user_id_col)[date_col].max().apply(lambda x: (snapshot_date - x).days)

    rfm = pd.DataFrame({"Recency": rec, "Frequency": freq, "Monetary": monetary})

    # ✅ Fix NaN trước khi phân loại
    rfm = rfm.fillna({"Recency": rfm["Recency"].max(), "Frequency": 0, "Monetary": 0})

    # ✅ Dùng cat.codes thay vì astype(int)
    rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1], duplicates="drop").cat.codes.replace(-1, 0)
    rfm["F"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4], duplicates="drop").cat.codes.replace(-1, 0)
    rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4], duplicates="drop").cat.codes.replace(-1, 0)

    return rfm.reset_index()

try:
    rfm_scores = compute_rfm_scores(df_trans)
except Exception as e:
    st.warning(f"Could not compute RFM automatically: {e}")
    rfm_scores = None

# ===============================
# Classification into 6 segments
# ===============================
def classify_customer(R, F, M):
    if R >= 3 and F >= 3 and M == 4:
        return """VIP - Khách hàng trung thành, giá trị cao 
    -> Thưởng bằng chương trình VIP, ưu đãi độc quyền 
    & Cho quyền truy cập sớm sản phẩm/dịch vụ mới 
    & Khuyến khích họ trở thành đại sứ thương hiệu"""
    elif F >= 3 and R >= 2 and M >= 2:
        return """Loyal Customers (Khách hàng trung thành tiềm năng) 
    -> Tăng cường gắn kết bằng ưu đãi định kỳ 
    & Cung cấp gói combo/bundles phù hợp 
    & Chăm sóc cá nhân hóa để đẩy lên nhóm VIP"""
    elif R == 2 and F <= 1 and M <= 1:
        return """At Risk/ Lost (Khách hàng có nguy cơ rời bỏ) 
    -> Giảm giá sản phẩm yêu thích trước đây 
    & Tìm hiểu lý do họ ít quay lại (khảo sát) 
    & Cung cấp ưu đãi đặc biệt để khuyến khích họ quay lại 
    & Tạo cảm giác cấp bách với ưu đãi giới hạn thời gian"""
    else:
        return """New/ Regular Customers (Khách hàng vãng lai) 
    -> Chiến dịch win-back với khuyến mãi lớn 
    & Tạo nội dung khuyến mãi để thu hút khách hàng mới 
    & Xin đánh giá sản phẩm/dịch vụ để cải thiện """

# ===============================
# Classification from raw values (not quartiles)
# ===============================
def classify_customer_raw(recency, frequency, monetary,
                          r_thresh, f_thresh, m_thresh):
    """
    Phân loại khách hàng dựa trên giá trị R, F, M gốc.
    - recency: số ngày kể từ lần mua gần nhất
    - frequency: số lần mua
    - monetary: tổng chi tiêu - (đơn vị: nghìn đồng)
    Ngưỡng có thể điều chỉnh theo dữ liệu thực tế.
    """
    if recency <= r_thresh/2 and frequency >= f_thresh and monetary >= m_thresh*1.5:
        return """VIP - Khách hàng trung thành, giá trị cao 
    -> Thưởng bằng chương trình VIP, ưu đãi độc quyền 
    & Cho quyền truy cập sớm sản phẩm/dịch vụ mới 
    & Khuyến khích họ trở thành đại sứ thương hiệu"""
    elif recency <= r_thresh and frequency >= f_thresh/2:
        return """Loyal Customers (Khách hàng trung thành tiềm năng) 
    -> Tăng cường gắn kết bằng ưu đãi định kỳ 
    & Cung cấp gói combo/bundles phù hợp 
    & Chăm sóc cá nhân hóa để đẩy lên nhóm VIP"""
    elif recency > r_thresh*3 and frequency <= f_thresh/2 and monetary <= m_thresh/2:
        return """At Risk/ Lost (Khách hàng có nguy cơ rời bỏ) 
    -> Giảm giá sản phẩm yêu thích trước đây 
    & Tìm hiểu lý do họ ít quay lại (khảo sát) 
    & Cung cấp ưu đãi đặc biệt để khuyến khích họ quay lại 
    & Tạo cảm giác cấp bách với ưu đãi giới hạn thời gian"""
    else:
        return """New/ Regular Customers (Khách hàng vãng lai) 
    -> Chiến dịch win-back với khuyến mãi lớn 
    & Tạo nội dung khuyến mãi để thu hút khách hàng mới 
    & Xin đánh giá sản phẩm/dịch vụ để cải thiện """

# Optimized clustering evaluation and comparison

def evaluate_clustering_performance(scaled_features, customer_data):
    """
    Evaluate clustering methods with Silhouette, ARI, NMI
    """
    clustering_methods = {
        "KMeans": "cluster_kmeans",
        "GMM": "cluster_gmm",
        "Agglomerative": "cluster_agg", 
        "Hierarchical": "cluster_hier",
        "SparkKMeans": "cluster_spark"
    }
    
    # --- Silhouette Scores ---
    silhouette_scores = {}
    for method_name, cluster_col in clustering_methods.items():
        if cluster_col in customer_data.columns:
            score = silhouette_score(scaled_features, customer_data[cluster_col])
            silhouette_scores[method_name] = round(score, 4)
    
    # --- Pairwise ARI & NMI ---
    similarity_results = []
    method_columns = list(clustering_methods.values())

    for i in range(len(method_columns)):
        for j in range(i+1, len(method_columns)):
            col1, col2 = method_columns[i], method_columns[j]
            if col1 in customer_data.columns and col2 in customer_data.columns:
                ari = adjusted_rand_score(customer_data[col1], customer_data[col2])
                nmi = normalized_mutual_info_score(customer_data[col1], customer_data[col2])
                similarity_results.append([
                    col1.replace('cluster_', '').title(),
                    col2.replace('cluster_', '').title(),
                    round(ari, 4),
                    round(nmi, 4)
                ])
    
    similarity_df = pd.DataFrame(similarity_results, columns=["Method A", "Method B", "ARI", "NMI"])
    return silhouette_scores, similarity_df


# ===============================
# Sidebar Menu
# ===============================
menu = st.sidebar.radio(
    "Menu",
    [
        "Introduction",
        "Business Problem",
        "Evaluation & Report & Comparison",
        "New Prediction / Analysis"
    ]
)
st.sidebar.header("Data")
product_up = st.sidebar.file_uploader("Upload Products_with_categories.csv", type=["csv"])
transaction_up = st.sidebar.file_uploader("Upload Transactions.csv", type=["csv"])
if product_up is None:
    df_products = pd.read_csv("Products_with_Categories.csv")
if transaction_up is None:
    df_trans = pd.read_csv("Transactions.csv")
st.sidebar.markdown("---")
# ===============================
# Sidebar Threshold Settings (for raw RFM)
# ===============================
st.sidebar.markdown("### ⚙️ Thiết lập ngưỡng RFM (Raw Values)")
r_thresh = st.sidebar.number_input("Ngưỡng Recency (ngày)", min_value=1, value=90)
f_thresh = st.sidebar.number_input("Ngưỡng Frequency (số lần mua)", min_value=1, value=10)
m_thresh = st.sidebar.number_input("Ngưỡng Monetary (tổng chi tiêu - đvt: nghìn đồng)", min_value=1, value=2000)
st.sidebar.markdown("---")

# ===============================
# Business Problem
# ===============================
if menu == "Business Problem":
    st.title("📌 Business Problem")
    st.markdown("""
    Cửa hàng X chủ yếu bán các sản phẩm thiết yếu cho khách hàng như:
    rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát...
    Khách hàng của cửa hàng thường là khách hàng mua lẻ.
    """)
    st.markdown("""-> Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn
    cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng,
    từ đó tăng doanh thu và lợi nhuận.
    Đồng thời, việc phân khúc khách hàng cũng giúp cửa hàng có chiến lược
    chăm sóc và làm hài lòng khách hàng.
    """)
    st.image('customersegment.png', caption='Customer Segmentation', width=800)

# ===============================
# Evaluation & Report
# ===============================
elif menu == "Evaluation & Report & Comparison":
    st.title("📊 Evaluation & Report & Comparison")
    st.image('segment1.png', caption='Customer Segment Pros', width=600)

    tab = st.tabs(["RFM Analysis, Segmentation & Visualization",
                "Clustering Model Comparison & Evaluation"])
    # ---------------- Tab 1 ----------------
    with tab[0]:
        # Ưu tiên dữ liệu upload
        if product_up is not None:
            df_products = pd.read_csv(product_up)
            st.success("✅ Products data uploaded successfully.")
            st.dataframe(df_products.head())
        else:
            df_products = df_products
            st.info("⚠️ Dùng dữ liệu Products mặc định.")

        if transaction_up is not None:
            df_trans = pd.read_csv(transaction_up)
            st.success("✅ Transactions data uploaded successfully.")
            st.dataframe(df_trans.head())
        else:
            df_trans = df_trans
            st.info("⚠️ Dùng dữ liệu Transactions mặc định.")

        # Tính toán RFM
        if rfm_scores is not None:
            rfm_scores = compute_rfm_scores(df_trans)
            rfm_scores["Segment_Raw"] = rfm_scores.apply(
                lambda x: classify_customer_raw(x["Recency"], x["Frequency"], x["Monetary"],
                                                r_thresh, f_thresh, m_thresh), axis=1
            )
            st.success("✅ RFM scores computed successfully.")
        # except Exception as e:
        #     st.error(f"❌ Lỗi khi tính RFM: {e}")
        #     rfm_scores = None

        # Nếu có RFM thì vẽ biểu đồ
        if rfm_scores is not None:
            st.subheader("📈 RFM Score Distribution")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.countplot(x="R", data=rfm_scores, ax=axes[0])
            axes[0].set_title("Recency Score")
            sns.countplot(x="F", data=rfm_scores, ax=axes[1])
            axes[1].set_title("Frequency Score")
            sns.countplot(x="M", data=rfm_scores, ax=axes[2])
            axes[2].set_title("Monetary Score")
            st.pyplot(fig)

            # Phân loại Segment
            rfm_scores["Segment"] = rfm_scores.apply(
                lambda x: classify_customer(x["R"], x["F"], x["M"]), axis=1
            )
            seg_counts = rfm_scores["Segment"].value_counts()

            st.bar_chart(seg_counts)
            st.dataframe(rfm_scores.head())
        else:
            st.warning("Không có dữ liệu RFM để hiển thị.")

    # ---------------- Tab 2 ----------------
    with tab[1]:
        st.subheader("Clustering Model Comparison")

    try:
        customer_data, scaled_features = customer_features(df_products, df_trans)
        silhouette_scores, similarity_df = evaluate_clustering_performance(scaled_features, customer_data)

        # Hiển thị Silhouette
        st.subheader("📊 Silhouette Scores")
        df_metrics = pd.DataFrame(list(silhouette_scores.items()), columns=["Model", "Silhouette"])
        st.table(df_metrics)

        # Hiển thị ARI & NMI
        st.subheader("🔗 Similarity Analysis (ARI & NMI)")
        st.dataframe(similarity_df)

        # Heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        pivot_table = similarity_df.pivot(index="Method A", columns="Method B", values="ARI")
        sns.heatmap(pivot_table, annot=True, cmap="Blues", ax=ax)
        ax.set_title("Heatmap of ARI")
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"❌ Lỗi khi tính toán clustering: {e}")
        st.markdown("""🔹 Kết quả so sánh nhanh
- KMeans & SparkKMeans:
Silhouette tốt nhất (≈0.285).
ARI & NMI gần 1.0 →
Hai phương pháp này gần như giống hệt nhau.
SparkKMeans có lợi thế về khả năng mở rộng (phù hợp khi dữ liệu lớn).
- Agglomerative & Hierarchical:
Kết quả giống hệt nhau (ARI = 1.0, NMI = 1.0).
Tuy nhiên, Silhouette thấp hơn (≈0.216).
Ưu điểm: trực quan hoá bằng dendrogram, dễ giải thích mối quan hệ phân cấp.
- GMM (Gaussian Mixture):
Silhouette thấp nhất (≈0.159).
ARI & NMI với các phương pháp khác đều thấp → tạo ra phân cụm rất khác biệt.
Chỉ phù hợp nếu muốn phân cụm mềm (soft clustering), tức một khách hàng có thể thuộc nhiều cụm.
-> 🔹 Khuyến nghị cuối cùng
- Chọn KMeans (hoặc SparkKMeans nếu dữ liệu lớn) làm phương pháp chính để phân cụm RFM.
Lý do: điểm Silhouette tốt nhất, tính ổn định cao, dễ áp dụng cho business.
SparkKMeans đặc biệt phù hợp nếu dữ liệu ngày càng mở rộng (big data).
- Dùng Agglomerative/Hierarchical như một phương pháp bổ trợ để kiểm tra lại kết quả và trực quan hóa mối quan hệ phân cụm.
- GMM có thể thử nghiệm nếu muốn kiểm tra xem khách hàng có thể thuộc nhiều nhóm đồng thời (phân tích nâng cao).
- Có thể dùng RFM Manual cho khả năng tương tác của end-user dễ hơn
""")

# ===============================
# New Prediction (manual input)
# ===============================
elif menu == "New Prediction / Analysis":
    st.title("🔮 Customer Prediction (Manual Input)")
    st.markdown("""
    - **VIP (High Frequency, High Monetary, Low Recency)** 
    -> Shop very often, spend a lot, and bought recently.
    - **Loyal Customer (Medium Frequency & Monetary, Recent Buyers)** 
    -> Bought recently, moderate spending, could become loyal. They’re on the path to becoming VIP.
    - **At Risk/ Lost Customer (Low Frequency, Low Monetary, High Recency)** 
    -> Send reminders, discounts to reactivate.  
    - **Regular/ New Customer (High Recency, Low Frequency & Monetary)** 
    -> Just purchased or bought only once. Still deciding whether to stick with you.
    """)
    tab = st.tabs(["RFM Manual Prediction", "KMeans Prediction (from Clustering)"])
    with tab[0]:
        st.subheader("RFM Manual Prediction")
        option = st.radio("Chọn kiểu nhập dữ liệu:", ["Nhập điểm RFM (1–4)", "Nhập giá trị gốc R, F, M"])
        if option == "Nhập điểm RFM (1–4)":
        # ---------------------------
        # Option 1: nhập điểm RFM
        # ---------------------------
                st.subheader("🔮 Dự đoán theo điểm RFM (1–4) - Single prediction")
                col1, col2, col3 = st.columns(3)
                with col1:
                    R = st.slider("Recency Score (1–4)", 1, 4, 2)
                with col2:
                    F = st.slider("Frequency Score (1–4)", 1, 4, 2)
                with col3:
                    M = st.slider("Monetary Score (1–4)", 1, 4, 2)

                if st.button("Predict by Score"):
                    segment = classify_customer(R, F, M)
                    st.success(f"🏷️ This customer belongs to: **{segment}**")

                st.markdown("---")
                st.subheader("📂 Bulk Prediction (File RFM Score)")
                file = st.file_uploader("Tải lên file CSV/Excel với cột `R`, `F`, `M`", type=["csv","xlsx"])
                if file:
                    if file.name.endswith(".csv"):
                        df_input = pd.read_csv(file)
                    else:
                        df_input = pd.read_excel(file)

                    if {"R","F","M"}.issubset(df_input.columns):
                        df_input["Segment"] = df_input.apply(lambda x: classify_customer(x["R"], x["F"], x["M"]), axis=1)
                        st.dataframe(df_input.head(20))
                        st.download_button(
                            "Tải xuống dự đoán (RFM Score)",
                            data=df_input.to_csv(index=False).encode("utf-8"),
                            file_name="rfm_predictions_score.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("File phải chứa các cột `R`, `F`, `M`.")

        # ---------------------------
        # Option 2: nhập giá trị gốc
        # ---------------------------
        else: 
            st.markdown("---")
            st.subheader("🔮 Dự đoán theo giá trị gốc")

            col1, col2, col3 = st.columns(3)
            with col1:
                recency_val = st.number_input("Recency (ngày kể từ lần mua gần nhất)", min_value=0, value=100)
            with col2:
                frequency_val = st.number_input("Frequency (số lần mua)", min_value=0, value=5)
            with col3:
                monetary_val = st.number_input("Monetary (tổng chi tiêu - đơn vị: nghìn đồng)", min_value=0, value=200)

            if st.button("Predict by Raw Value"):
                segment = classify_customer_raw(recency_val, frequency_val, monetary_val,
                                                        r_thresh=r_thresh, f_thresh=f_thresh, m_thresh=m_thresh)
                st.success(f"🏷️ This customer belongs to: **{segment}**")

            st.markdown("---")
            st.subheader("📂 Bulk Prediction (File Raw Values)")
            file = st.file_uploader("Tải lên file CSV/Excel với cột `R - Recency`, `F - Frequency`, `M - Monetary`", type=["csv","xlsx"])
            if file:
                if file.name.endswith(".csv"):
                    df_input = pd.read_csv(file)
                else:
                    df_input = pd.read_excel(file)

                if {"Recency","Frequency","Monetary"}.issubset(df_input.columns):
                    df_input["Segment"] = df_input.apply(
                        lambda x: classify_customer_raw(
                            x["Recency"], x["Frequency"], x["Monetary"],
                            r_thresh=r_thresh, f_thresh=f_thresh, m_thresh=m_thresh
                            ),
                        axis=1
                        )
                    st.dataframe(df_input.head(20))
                    st.download_button(
                        "Tải xuống dự đoán (Raw Values)",
                        data=df_input.to_csv(index=False).encode("utf-8"),
                        file_name="rfm_predictions_raw.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("File phải chứa các cột `R`, `F`, `M`.")
        with tab[1]:
            st.title("💡 Recommendations by KMeans Clusters")
            try:
                # chạy lại phân cụm KMeans
                customer_data, scaled_features = customer_features(df_products, df_trans)

                # Đếm số khách hàng theo cụm
                cluster_counts = customer_data["cluster_kmeans"].value_counts().sort_index()

                st.subheader("📊 Phân bố khách hàng theo cụm (KMeans)")
                st.bar_chart(cluster_counts)

                # Hiển thị bảng khách hàng mẫu
                st.dataframe(customer_data[["Member_number","Recency","Frequency","Monetary","cluster_kmeans"]].head())

                # Gợi ý chiến lược theo từng cụm
                st.subheader("💡 Gợi ý chiến lược kinh doanh")

                recommendations = {
                    0: "Cluster 0 – Có thể là khách mới: Khuyến mãi chào mừng, ưu đãi lần mua đầu.",
                    1: "Cluster 1 – Khách trung thành giá trị cao: Chương trình VIP, chăm sóc cá nhân hóa.",
                    2: "Cluster 2 – Khách ít tương tác, chi tiêu thấp: Gửi chiến dịch tái kích hoạt, giảm giá.",
                    3: "Cluster 3 – Khách chi tiêu nhiều nhưng ít mua: Khuyến khích mua thường xuyên, gợi ý combo."
                }

                for cluster, note in recommendations.items():
                    st.markdown(f"**Cụm {cluster}:** {note}")

            except Exception as e:
                st.error(f"❌ Lỗi khi tạo recommendation từ KMeans: {e}")
# ===============================
# Introduction
# ===============================
elif menu == "Introduction":
    st.title("👨‍💻 Introduction")
    st.markdown("""
    - **Tên**: Trần Nhật Minh   
    - **Email**: nhatminhtr233@gmail.com   
    - **GVHD**: Khuất Thuỳ Phương
    - **Project**: Customer Segmentation
    """)
    st.image("RFM_clustering.png", caption="RFM Clustering")
    st.subheader("🔄 Project Pipeline")
    st.markdown("""
        - Business Problem", "Xác định mục tiêu kinh doanh, ví dụ: tăng doanh thu, chăm sóc khách hàng.
        - Data Preparation", "Thu thập & làm sạch dữ liệu sản phẩm và giao dịch.
        - RFM Analysis", "Tính toán Recency, Frequency, Monetary cho từng khách hàng.
        - Clustering Models", "Thử nhiều mô hình: KMeans, GMM, Agglomerative, Hierarchical, SparkKMeans.
        - Evaluation", "Đánh giá bằng Silhouette, ARI, NMI để so sánh mô hình.
        - Recommendation & Deployment", "Đưa ra gợi ý kinh doanh cho từng nhóm khách hàng và triển khai hệ thống."""
    )
    st.image('pipeline.png', caption = 'Project Pipeline')
