# 🖥 Backend – Smart Dataset Analyzer  

The **backend** is built with **FastAPI** and provides NLP/ML-powered **text analysis APIs** and **PDF export**.  

---

## 🚀 Features  
- **Text Preprocessing** → tokenization, stop-word removal, lemmatization  
- **Topic Modeling** → LDA, NMF  
- **Sentiment Analysis** → TextBlob  
- **Vectorization** → TF-IDF, BoW  
- **PDF Export** → auto-generated summary report  

---

## 📂 Structure  
backend/
│── analyze_all.py # FastAPI main app
│── requirements.txt # Dependencies
│── README.md

yaml
Copy code

Endpoints:  
- `POST /analyze-all` → Upload + analyze dataset  
- `POST /download-report` → Export PDF  

---

## 🛠 Setup & Run  

### Install dependencies  
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Start server
bash
Copy code
uvicorn analyze_all:app --reload --host 127.0.0.1 --port 8001
Server runs at → http://localhost:8001/

📄 Requirements
Python 3.8+

FastAPI, Uvicorn, scikit-learn, TextBlob, SpaCy, Pandas, NumPy, python-docx, fpdf

Example (requirements.txt):

txt
Copy code
fastapi
uvicorn
pandas
numpy
scikit-learn
textblob
spacy
python-docx
fpdf
yaml
Copy code

---

👉 This way, your repo has:  

- `README.md` → overview + project structure  
- `frontend/README.md` → React/Tailwind details  
- `backend/README.md` → FastAPI/NLP details  