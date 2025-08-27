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
  <img src="https://raw.githubusercontent.com/ssk-2003/Smart-Dataset-Analyzer/main/frontend/public/assets/landing.png" alt="Landing Page" width="300"/>
  <img src="https://raw.githubusercontent.com/ssk-2003/Smart-Dataset-Analyzer/main/frontend/public/assets/upload.png" alt="Upload Page" width="300"/>
  <img src="https://raw.githubusercontent.com/ssk-2003/Smart-Dataset-Analyzer/main/frontend/public/assets/results.png" alt="Analysis Results" width="300"/>
</div>


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
---

### 2. Backend Setup (FastAPI)
```bash
cd backend
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn analyze_all:app --reload --port 8001
Backend will be live at 👉 http://127.0.0.1:8001
3. Frontend Setup (React)
cd frontend
npm install
npm start


Frontend will be live at 👉 http://localhost:3000

📂 Project Structure
Smart-Dataset-Analyzer/
├── backend/
│   ├── analyze_all.py          # FastAPI backend API
│   ├── requirements.txt        # Python backend dependencies
│   └── utils/                  # (optional helper scripts)
│
├── frontend/
│   ├── public/                 # Public assets (favicon, index.html, screenshots)
│   │   └── assets/             # Screenshots for README
│   │       ├── landing.png
│   │       ├── upload.png
│   │       └── results.png
│   ├── src/
│   │   ├── components/         # Reusable React components
│   │   ├── pages/              # Main app pages
│   │   ├── App.js / App.css    # Root app shell
│   │   └── index.js            # Entry point
│   ├── package.json
│   └── README.md               # Frontend-specific docs
│
├── .gitignore
└── README.md                   # 👉 Main documentation

📊 Example Workflow

Upload .txt, .csv, or .docx file

Backend extracts text and runs NLP pipeline

Topics & sentiments are detected and summarized

Frontend displays insights, charts, and top terms

Export full report as PDF 📄

🤝 Contributing

Fork the repo

Create a branch (git checkout -b feature/xyz)

Commit your changes (git commit -m "Add new feature")

Push to branch (git push origin feature/xyz)

Open a Pull Request

📜 License

MIT License © 2025 Satish Kumar

⭐ Support

If you like this project, give it a ⭐ on GitHub
!


---

✅ Just replace your current `README.md` with this version, commit, and push:  

```bash
git add README.md
git commit -m "Complete README with setup and documentation"
git push origin main
