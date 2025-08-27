# 🚀 Smart Dataset Analyzer

**AI-powered text analysis platform** — upload `.txt`, `.csv`, or `.docx` files to instantly get **sentiment analysis**, **topic modeling**, **key terms**, and **downloadable PDF reports**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-blue?logo=react)
![Tailwind](https://img.shields.io/badge/TailwindCSS-3.x-38B2AC?logo=tailwind-css)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- ✅ **spaCy-powered NLP** → tokenization, lemmatization, stop-word removal  
- ✅ **Sentiment Analysis** → Positive, Negative, Neutral breakdown  
- ✅ **Topic Modeling** → NMF (default) and LDA  
- ✅ **Top Terms Extraction** → for word clouds & quick insights  
- ✅ **Smart Recommendations** → based on dataset sentiment & topics  
- ✅ **PDF Report Export** → summary, insights, topics, and sentiment  

---

## 🖼️ UI Preview

<div align="center">
  <img src="frontend/public/assets/landing.png" alt="Landing Page" width="300"/>
  <img src="frontend/public/assets/upload.png" alt="Upload Page" width="300"/>
  <img src="frontend/public/assets/results.png" alt="Analysis Results" width="300"/>
</div>

> ⚡ Place your actual screenshots in `frontend/public/assets/` with names `landing.png`, `upload.png`, `results.png`.

---

## 🛠️ Tech Stack

| Layer       | Technologies                                                                 |
|-------------|-------------------------------------------------------------------------------|
| **Backend** | FastAPI, spaCy, scikit-learn, pandas, TextBlob, ReportLab, FPDF              |
| **Frontend**| React, Tailwind CSS, Framer Motion, recharts, react-dropzone                 |
| **Reporting** | PDF generation with ReportLab + FPDF                                       |

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ssk-2003/Smart-Dataset-Analyzer.git
cd Smart-Dataset-Analyzer
