"""Dashboard StudentDOA: Data -> Business Intelligence -> ML -> LLM."""
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import BASE_DIR, FEATURES_BY_CUTOFF, TARGET
from vai_analyst import DEFAULT_MODEL, build_prompt, generate_student_analysis

DATA_PATH = BASE_DIR / "studentDOA.csv"

st.set_page_config(page_title="StudentDOA — Data, ML dan LLM", page_icon="🎓", layout="wide")
st.title("StudentDOA — dari Data ke ML dan LLM")
st.caption("Dashboard pembelajaran untuk memahami status dan perkembangan mahasiswa.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(cutoff):
    path = BASE_DIR / f"student_outcome_{cutoff.lower().replace(' ', '')}.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_model_outputs(cutoff):
    suffix = cutoff.lower().replace(" ", "")
    metrics_path = BASE_DIR / "model_metrics.json"
    importance_path = BASE_DIR / f"feature_importance_{suffix}.csv"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    return all_metrics.get(cutoff, {}), importance


try:
    students = load_data()
except Exception as exc:
    st.error(f"Dataset gagal dimuat: {exc}")
    st.stop()

st.sidebar.header("Pilih catatan mahasiswa")
selected_number = st.sidebar.selectbox("Nomor baris", range(1, len(students) + 1), index=0)
cutoff = st.sidebar.radio("Data tersedia hingga", ["Semester 1", "Semester 2"], index=0)
student = students.iloc[selected_number - 1]
features = FEATURES_BY_CUTOFF[cutoff]
X_student = student[features].to_frame().T
model = load_model(cutoff)
metrics, feature_importance = load_model_outputs(cutoff)

status_counts = students[TARGET].value_counts(normalize=True) * 100
m1, m2, m3, m4 = st.columns(4)
m1.metric("Catatan mahasiswa", f"{len(students):,}")
m2.metric("Dropout", f"{status_counts.get('Dropout', 0):.1f}%")
m3.metric("Graduate", f"{status_counts.get('Graduate', 0):.1f}%")
m4.metric("Enrolled", f"{status_counts.get('Enrolled', 0):.1f}%")

tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · Data", "2 · Business Intelligence", "3 · Machine Learning", "4 · LLM Analyst",
])

with tab_data:
    st.subheader("Profil, pendaftaran, keuangan, akademik, dan ekonomi")
    st.write(f"**{len(students):,} baris × {len(students.columns)} kolom.** Target terdiri dari tiga kelas.")
    st.caption(f"Tampilan fitur dibatasi pada data yang tersedia hingga {cutoff}.")
    display_columns = features + [TARGET]
    selected_record = student[display_columns].to_frame().T
    selected_record.insert(0, "Nomor baris", selected_number)
    st.dataframe(selected_record, use_container_width=True, hide_index=True)
    with st.expander("Lihat beberapa baris dataset"):
        preview = students[display_columns].head(100).copy()
        preview.insert(0, "Nomor baris", range(1, len(preview) + 1))
        st.dataframe(preview, use_container_width=True)
    st.info("Sebagian variabel, seperti Course dan kualifikasi orang tua, adalah kode kategori. Perlakukan sebagai kategori, bukan skala angka.")

with tab_bi:
    st.subheader("Business Intelligence")
    st.markdown("**Gambaran umum mahasiswa**")
    b1, b2, b3 = st.columns(3)
    b1.metric("Rata-rata usia saat mendaftar", f"{students['Age at enrollment'].mean():.1f} tahun")
    b2.metric("Pemegang beasiswa", f"{students['Scholarship holder'].mean():.1%}")
    b3.metric("Biaya kuliah tercatat lunas", f"{students['Tuition fees up to date'].mean():.1%}")

    left_status, right_status = st.columns(2)
    with left_status:
        st.markdown("**Jumlah mahasiswa menurut status**")
        st.bar_chart(students[TARGET].value_counts().rename_axis("Status").to_frame("Jumlah mahasiswa"))
    with right_status:
        st.markdown("**Proporsi status menurut kelompok usia**")
        age_bins = [0, 20, 25, 30, 40, 100]
        age_labels = ["≤20", "21–25", "26–30", "31–40", ">40"]
        age_data = students[["Age at enrollment", TARGET]].copy()
        age_data["Kelompok usia"] = pd.cut(age_data["Age at enrollment"], bins=age_bins, labels=age_labels)
        age_mix = pd.crosstab(age_data["Kelompok usia"], age_data[TARGET], normalize="index").mul(100)
        st.bar_chart(age_mix)

    left, right = st.columns(2)
    semester_columns = ["1st sem"] if cutoff == "Semester 1" else ["1st sem", "2nd sem"]
    approved_columns = [f"Curricular units {semester} (approved)" for semester in semester_columns]
    grade_columns = [f"Curricular units {semester} (grade)" for semester in semester_columns]
    approved_labels = {column: f"Semester {index + 1}" for index, column in enumerate(approved_columns)}
    grade_labels = {column: f"Semester {index + 1}" for index, column in enumerate(grade_columns)}
    with left:
        st.markdown("**Rata-rata mata kuliah yang disetujui**")
        approved = students.groupby(TARGET)[approved_columns].mean().rename(columns=approved_labels)
        st.bar_chart(approved)
    with right:
        st.markdown("**Rata-rata nilai semester**")
        grades = students.groupby(TARGET)[grade_columns].mean().rename(columns=grade_labels)
        st.bar_chart(grades)

    st.markdown("**Proporsi status berdasarkan kondisi pendaftaran**")
    factors = [
        ("Scholarship holder", "Status beasiswa (0 = tidak, 1 = ya)"),
        ("Tuition fees up to date", "Status pembayaran biaya kuliah (0 = belum, 1 = lunas)"),
        ("Debtor", "Status tunggakan (0 = tidak, 1 = ya)"),
    ]
    factor_cols = st.columns(len(factors))
    for column, (field, label) in zip(factor_cols, factors):
        with column:
            st.markdown(f"**{label}**")
            mix = pd.crosstab(students[field], students[TARGET], normalize="index").mul(100)
            mix.index = [str(int(value)) for value in mix.index]
            st.bar_chart(mix)

    st.markdown("**10 kode program studi dengan mahasiswa terbanyak**")
    top_courses = students["Course"].value_counts().head(10).index
    course_data = students[students["Course"].isin(top_courses)]
    course_mix = pd.crosstab(course_data["Course"], course_data[TARGET], normalize="index").mul(100)
    st.bar_chart(course_mix.sort_index())
    st.caption("Grafik memperlihatkan pola dalam dataset; bukan bukti bahwa usia, beasiswa, pembayaran, atau program studi menyebabkan suatu status.")
    st.caption("Nilai 0 bisa menunjukkan belum mengikuti evaluasi. Pertimbangkan kolom evaluations saat membaca rata-rata nilai.")

with tab_ml:
    st.subheader(f"Prediksi status mahasiswa — data hingga {cutoff}")
    if model is None:
        st.warning("Model belum tersedia. Jalankan `python data_prep.py`, lalu `python ml_model.py` dari folder StudentDOA.")
    else:
        probabilities = dict(zip(model.classes_, model.predict_proba(X_student)[0]))
        predicted = max(probabilities, key=probabilities.get)
        a, b = st.columns(2)
        a.metric("Prediksi status", str(predicted))
        b.metric("Probabilitas tertinggi", f"{probabilities[predicted]:.1%}")
        st.markdown("**Probabilitas tiap status**")
        st.bar_chart(pd.Series(probabilities, name="Probabilitas"))
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            c3.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
            st.caption("Metrik dihitung pada data test yang tidak digunakan saat pelatihan.")
        if not feature_importance.empty:
            st.markdown("**Fitur yang paling berpengaruh secara keseluruhan**")
            st.bar_chart(feature_importance.head(15).set_index("feature")["importance"])
        st.warning("Model Semester 1 hanya menggunakan informasi hingga semester 1. Model Semester 2 memakai informasi hingga semester 2. Prediksi merupakan perkiraan, bukan kepastian atau bukti sebab-akibat.")

with tab_llm:
    st.subheader("LLM Analyst — interpretasi berbasis data")
    st.caption(f"Ollama model: {DEFAULT_MODEL}")
    if model is None:
        st.info("Jalankan persiapan data dan pelatihan model terlebih dahulu.")
    else:
        probabilities = dict(zip(model.classes_, model.predict_proba(X_student)[0]))
        factors = feature_importance.head(8).to_dict("records") if not feature_importance.empty else []
        profile = student[features + [TARGET]].to_dict()
        prompt = build_prompt(profile, probabilities, factors)
        with st.expander("Lihat prompt yang dikirim ke LLM"):
            st.code(prompt, language="text")
        if st.button("Buat analisis mahasiswa"):
            with st.spinner("Meminta Ollama menganalisis profil dan prediksi..."):
                st.markdown(generate_student_analysis(profile, probabilities, factors))

st.markdown("---")
st.caption("Alur: dataset → business intelligence → prediksi machine learning → interpretasi LLM.")
    
