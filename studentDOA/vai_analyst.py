"""Optional Ollama LLM layer for grounded student outcome interpretation."""
import os

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(student_profile: dict, probabilities: dict, model_factors: list | None = None) -> str:
    return f"""Kamu adalah asisten analisis data pendidikan. Jawab dalam bahasa Indonesia.
Gunakan hanya informasi yang tersedia. Prediksi adalah perkiraan model, bukan kepastian
atau bukti penyebab mahasiswa akan dropout. Jangan menyimpulkan kondisi personal yang
tidak ada di data. Hindari rekomendasi yang menghakimi.

PROFIL MAHASISWA (kolom dataset)
{student_profile}

PROBABILITAS STATUS HASIL MODEL
{probabilities}

FITUR GLOBAL YANG PALING DIPAKAI MODEL
{model_factors or 'Tidak tersedia'}

Buat ringkasan singkat: interpretasikan prediksi, sebutkan dua hal yang benar-benar
terlihat pada data, berikan dua bentuk dukungan akademik yang wajar, dan jelaskan satu
batasan prediksi. Ingat bahwa banyak kolom kategori berupa kode angka."""


def generate_student_analysis(student_profile: dict, probabilities: dict, model_factors=None, model=DEFAULT_MODEL) -> str:
    prompt = build_prompt(student_profile, probabilities, model_factors)
    try:
        import ollama
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
        return response["message"]["content"]
    except Exception as exc:
        return f"Tidak dapat menghubungi Ollama model '{model}'. Pastikan Ollama aktif dan model tersedia. Detail: {exc}"
