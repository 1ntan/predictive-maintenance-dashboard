import streamlit as st
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("⚙️ Deep Learning untuk Predictive Maintenance pada IIoT")

st.markdown("""
### Prediksi Remaining Useful Life (RUL) Menggunakan Deep Learning

**Domain:** Industrial Internet of Things (IIoT)

**Dataset:** NASA C-MAPSS FD001
""")

st.info(
    "Proyek ini bertujuan memprediksi Remaining Useful Life (RUL) mesin turbofan menggunakan model Deep Learning berbasis data sensor multivariat."
)

# =====================================================
# KPI CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Unit Mesin",
    "100"
)

col2.metric(
    "Sensor",
    "21"
)

col3.metric(
    "RMSE Terbaik",
    "13.24"
)

col4.metric(
    "MAE Terbaik",
    "9.68"
)

st.divider()

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Ringkasan",
    "📂 Dataset",
    "⚙️ Preprocessing",
    "📊 Perbandingan Model",
    "📈 Hasil",
    "🏆 Kesimpulan"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.header("Ringkasan Proyek")

    st.write("""
    Kerusakan mesin yang tidak terdeteksi dapat menyebabkan downtime produksi,
    peningkatan biaya perawatan, serta menurunkan efisiensi operasional.

    Pada proyek ini dilakukan pengembangan model Deep Learning
    untuk memprediksi Remaining Useful Life (RUL)
    menggunakan dataset NASA C-MAPSS FD001.
    """)

    st.subheader("Tujuan Penelitian")

    st.write("""
    Mengembangkan model prediksi Remaining Useful Life (RUL)
    yang akurat untuk mendukung implementasi Predictive Maintenance
    pada lingkungan Industrial Internet of Things (IIoT).
    """)

    st.subheader("Model yang Digunakan")

    st.markdown("""
    - 1D-CNN
    - CNN-LSTM
    - Transformer
    """)

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.header("Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### NASA C-MAPSS FD001

        - 100 Unit Mesin
        - 21 Sensor Operasional
        - 3 Variabel Kondisi Operasi
        - Data Multivariat Time-Series
        - Target: Remaining Useful Life (RUL)
        """)

    with col2:

        st.success("""
        Subset FD001 Dipilih

        • Kondisi Operasi Tunggal

        • Banyak digunakan pada penelitian Predictive Maintenance

        • Cocok untuk benchmarking model Deep Learning
        """)

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.header("Tahapan Preprocessing Data")

    st.markdown("""
    ### Tahapan yang Dilakukan

    ✅ Piece-wise RUL Labeling

    ✅ Feature Selection

    ✅ Normalisasi MinMax

    ✅ Signal Denoising

    ✅ Sequence Generation

    ✅ Sliding Window (30 Siklus)
    """)

    st.subheader("Visualisasi Signal Denoising")

    st.image(
        "assets/denoising_signal.png",
        use_container_width=True
    )

    st.write("""
    Teknik Moving Average digunakan untuk mengurangi noise pada sinyal sensor.
    Hasil denoising menghasilkan pola degradasi mesin yang lebih stabil
    sehingga dapat meningkatkan kualitas data sebelum proses pelatihan model.
    """)

# =====================================================
# TAB 4
# =====================================================

with tab4:

    st.header("Perbandingan Performa Model")

    comparison = pd.DataFrame({

        "Model": [
            "1D-CNN",
            "CNN-LSTM",
            "Transformer"
        ],

        "RMSE": [
            14.46,
            13.24,
            16.54
        ],

        "MAE": [
            11.00,
            9.68,
            12.10
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Perbandingan RMSE")

        st.bar_chart(
            comparison.set_index("Model")["RMSE"]
        )

    with col2:

        st.subheader("Perbandingan MAE")

        st.bar_chart(
            comparison.set_index("Model")["MAE"]
        )

    st.success("""
    🏆 Model Terbaik: CNN-LSTM

    RMSE : 13.24

    MAE : 9.68
    """)

# =====================================================
# TAB 5
# =====================================================

with tab5:

    st.header("Hasil Model CNN-LSTM")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "assets/cnn_lstm_loss.png",
            caption="Training Loss CNN-LSTM",
            use_container_width=True
        )

    with col2:

        st.image(
            "assets/cnn_lstm_prediction.png",
            caption="Prediksi CNN-LSTM",
            use_container_width=True
        )

    st.write("""
    CNN-LSTM menggabungkan Conv1D untuk ekstraksi fitur lokal
    dan LSTM untuk mempelajari hubungan temporal pada data sensor.
    """)

    st.divider()

    st.header("Hasil Model Transformer")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "assets/transformer_loss.png",
            caption="Training Loss Transformer",
            use_container_width=True
        )

    with col2:

        st.image(
            "assets/transformer_prediction.png",
            caption="Prediksi Transformer",
            use_container_width=True
        )

    st.write("""
    Transformer menggunakan mekanisme Self-Attention
    untuk menangkap hubungan jangka panjang antar data sensor.
    """)

# =====================================================
# TAB 6
# =====================================================

with tab6:

    st.header("Kesimpulan")

    st.success("""
    CNN-LSTM memberikan performa terbaik pada dataset NASA C-MAPSS FD001.
    """)

    st.subheader("Ranking Model")

    ranking = pd.DataFrame({
        "Peringkat": ["🥇", "🥈", "🥉"],
        "Model": [
            "CNN-LSTM",
            "1D-CNN",
            "Transformer"
        ]
    })

    st.table(ranking)

    st.subheader("Hasil Akhir")

    st.markdown("""
    **CNN-LSTM**

    - RMSE = 13.24
    - MAE = 9.68
    """)

    st.subheader("Temuan Penelitian")

    st.markdown("""
    - CNN-LSTM menghasilkan akurasi prediksi terbaik.
    - Conv1D efektif dalam mengekstraksi pola degradasi mesin.
    - LSTM mampu menangkap hubungan temporal pada data sensor.
    - Transformer masih memerlukan tuning lebih lanjut untuk meningkatkan performa.
    """)

    st.divider()

    st.subheader("Informasi Proyek")

    st.markdown("""
    **Judul Proyek**

    Deep Learning untuk Predictive Maintenance pada Industrial Internet of Things (IIoT)

    **Dataset**

    NASA C-MAPSS FD001

    **Capstone Project DSC 2026**
    """)
