import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="K-Means Gempa Bumi Global", page_icon="🌍", layout="wide")

st.title("🌍 Analisis Clustering Gempa Bumi Global dengan K-Means")
st.write(
    "Dashboard interaktif untuk mengeksplorasi data gempa, melakukan standardisasi fitur, "
    "menentukan jumlah cluster, mengevaluasi model, dan memahami karakteristik setiap cluster."
)

DEFAULT_PATH = "global_earthquakes_seismic.csv"
uploaded = st.sidebar.file_uploader("Unggah dataset CSV (opsional)", type=["csv"])

@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = "File yang diunggah"
    elif os.path.exists(DEFAULT_PATH):
        df = load_csv_from_path(DEFAULT_PATH)
        data_source = DEFAULT_PATH
    else:
        st.info("Unggah file CSV dataset gempa melalui sidebar, atau letakkan global_earthquakes_seismic.csv di folder app.py.")
        st.stop()
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

st.caption(f"Sumber data aktif: {data_source}")
st.subheader("1. Ringkasan dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah baris", f"{len(df):,}")
c2.metric("Jumlah kolom", f"{df.shape[1]:,}")
c3.metric("Kolom dengan missing value", int(df.isna().any().sum()))

with st.expander("Lihat sampel dan tipe data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    dtype_df = pd.DataFrame({"kolom": df.columns, "tipe_data": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True)
    st.write("Ringkasan statistik numerik")
    st.dataframe(df.describe().T, use_container_width=True)
    st.write("Missing values per kolom")
    st.dataframe(df.isna().sum().rename("missing_values").to_frame(), use_container_width=True)

# Kandidat fitur numerik yang relevan jika tersedia.
candidate_features = [
    "latitude", "longitude", "depth", "mag", "sig",
    "tsunami", "nst", "dmin", "rms", "gap", "confidence_score"
]
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
default_features = [c for c in candidate_features if c in numeric_columns]
if len(default_features) < 2:
    default_features = numeric_columns[:min(5, len(numeric_columns))]

st.sidebar.header("Pengaturan clustering")
selected_features = st.sidebar.multiselect(
    "Pilih fitur numerik untuk clustering",
    options=numeric_columns,
    default=default_features
)

if len(selected_features) < 2:
    st.warning("Pilih minimal dua fitur numerik di sidebar agar clustering dapat dijalankan.")
    st.stop()

# Hindari timestamp mentah dan kolom ID sebagai fitur kecuali pengguna memilihnya secara eksplisit.
X_raw = df[selected_features].copy()
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
for col in X_raw.columns:
    median_value = X_raw[col].median()
    X_raw[col] = X_raw[col].fillna(median_value if pd.notna(median_value) else 0)

# Hapus duplikasi pada fitur yang dipilih untuk mencegah baris identik mendominasi.
X_raw = X_raw.drop_duplicates().reset_index(drop=True)
if len(X_raw) < 3:
    st.error("Data bersih terlalu sedikit. Pilih fitur lain atau periksa dataset.")
    st.stop()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=selected_features)

st.subheader("2. Eksplorasi data (EDA)")
st.write("Distribusi fitur membantu melihat bentuk sebaran dan kemungkinan nilai ekstrem.")
dist_feature = st.selectbox("Pilih fitur untuk histogram", selected_features, key="dist")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(X_raw[dist_feature], kde=True, ax=ax)
ax.set_title(f"Distribusi {dist_feature}")
st.pyplot(fig)
plt.close(fig)
st.caption(
    f"Interpretasi: histogram menunjukkan sebaran nilai {dist_feature}. Kemencengan atau ekor panjang "
    "dapat mengindikasikan distribusi tidak simetris; nilai ekstrem perlu ditinjau, tetapi tidak otomatis dihapus."
)

st.write("Korelasi antarfitur")
fig, ax = plt.subplots(figsize=(max(8, len(selected_features)), max(5, len(selected_features) * 0.55)))
sns.heatmap(X_raw.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Interpretasi: korelasi mendekati +1 menunjukkan hubungan linear positif, sedangkan mendekati -1 "
    "menunjukkan hubungan linear negatif. Korelasi bukan bukti sebab-akibat."
)

with st.expander("Lihat boxplot semua fitur", expanded=False):
    ncols = 3
    nrows = int(np.ceil(len(selected_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(selected_features):
        sns.boxplot(x=X_raw[col], ax=axes[i])
        axes[i].set_title(col)
    for j in range(len(selected_features), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Interpretasi: titik/ekor yang jauh dari rentang utama dapat menandakan nilai ekstrem. Validitasnya perlu dilihat dari konteks data gempa.")

st.subheader("3. Elbow Method dan Silhouette Score")
max_k = min(10, len(X_scaled) - 1)
if max_k < 2:
    st.error("Jumlah data tidak cukup untuk menghitung metrik clustering.")
    st.stop()

k_values = list(range(2, max_k + 1))
inertias, silhouettes = [], []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

metrics_df = pd.DataFrame({
    "K": k_values,
    "Inertia": inertias,
    "Silhouette Score": silhouettes
})
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("Jumlah cluster (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Interpretasi: inertia selalu cenderung turun saat K bertambah. Cari titik siku ketika penurunannya mulai melambat.")
with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, silhouettes, marker="o")
    ax.set_xlabel("Jumlah cluster (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Interpretasi: skor mendekati 1 menunjukkan cluster lebih terpisah; sekitar 0 menunjukkan tumpang tindih; nilai negatif dapat menandakan penempatan yang kurang sesuai.")

st.dataframe(metrics_df.round(4), use_container_width=True)
recommended_k = int(metrics_df.loc[metrics_df["Silhouette Score"].idxmax(), "K"])
st.info(f"K dengan Silhouette Score tertinggi dalam rentang yang diuji: {recommended_k}. Pertimbangkan juga titik siku pada grafik.")

k = st.sidebar.slider("Jumlah cluster K", min_value=2, max_value=max_k, value=recommended_k)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, labels)
inertia = float(kmeans.inertia_)

st.subheader("4. Hasil clustering")
m1, m2, m3 = st.columns(3)
m1.metric("K yang dipilih", k)
m2.metric("Inertia", f"{inertia:,.2f}")
m3.metric("Silhouette Score", f"{sil:.3f}")

result = X_raw.copy()
result["cluster"] = labels

st.write("Pilih dua fitur untuk visualisasi scatter plot")
plot_x = st.selectbox("Sumbu X", selected_features, index=selected_features.index("longitude") if "longitude" in selected_features else 0)
default_y_idx = selected_features.index("latitude") if "latitude" in selected_features and selected_features.index("latitude") != selected_features.index(plot_x) else (1 if selected_features[0] == plot_x else 0)
plot_y_options = [x for x in selected_features if x != plot_x]
plot_y = st.selectbox("Sumbu Y", plot_y_options, index=0)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=result, x=plot_x, y=plot_y, hue="cluster", palette="tab10", alpha=0.75, s=35, ax=ax)
ax.set_title(f"Cluster K-Means: {plot_x} vs {plot_y}")
st.pyplot(fig)
plt.close(fig)
st.caption(
    f"Interpretasi: setiap warna mewakili cluster berdasarkan semua fitur terpilih, sedangkan grafik menampilkan "
    f"hanya {plot_x} dan {plot_y}. Karena itu, kelompok bisa tampak tumpang tindih pada plot 2D walaupun dibentuk dari banyak fitur."
)

st.subheader("5. Profil karakteristik cluster")
profile = result.groupby("cluster")[selected_features].mean().round(3)
sizes = result["cluster"].value_counts().sort_index().rename("jumlah_data")
profile_display = profile.copy()
profile_display.insert(0, "jumlah_data", sizes)
st.dataframe(profile_display, use_container_width=True)

fig, ax = plt.subplots(figsize=(max(9, len(selected_features)), 5))
profile.T.plot(kind="bar", ax=ax)
ax.set_title("Rata-rata fitur per cluster (skala asli)")
ax.set_ylabel("Rata-rata")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Cluster")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Interpretasi: bandingkan rata-rata antarcluster untuk mengetahui fitur yang paling membedakan kelompok. "
    "Karena fitur bisa memiliki satuan berbeda, perbandingan tinggi batang antarfitur tidak selalu sebanding."
)

with st.expander("Lihat median per cluster", expanded=False):
    st.dataframe(result.groupby("cluster")[selected_features].median().round(3), use_container_width=True)

st.subheader("6. Unduh hasil")
normalized_df = pd.DataFrame(X_scaled, columns=selected_features)
normalized_df["cluster"] = labels
st.download_button(
    "Unduh data fitur ternormalisasi + cluster (CSV)",
    data=normalized_df.to_csv(index=False).encode("utf-8"),
    file_name="adli nabil mubarok_Test3_ClusteringK-Means_normalized.csv",
    mime="text/csv"
)
st.download_button(
    "Unduh hasil clustering pada skala asli (CSV)",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="adli nabil mubarok_Test3_ClusteringK-Means_hasil.csv",
    mime="text/csv"
)
st.download_button(
    "Unduh metrik evaluasi untuk setiap K (CSV)",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="adli nabil mubarok_Test3_ClusteringK-Means_metrics.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "Catatan metodologis: hasil K-Means sensitif terhadap fitur dan skala, sehingga fitur distandardisasi. "
    "Label cluster adalah hasil pengelompokan matematis dan tidak boleh langsung dianggap sebagai kategori bahaya gempa yang tervalidasi."
)