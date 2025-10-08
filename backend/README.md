🖥 Smart NLP Text Analytics Engine – Backend

The backend of Smart NLP Text Analytics Engine is powered by FastAPI, providing advanced NLP and ML-driven text analysis with professional PDF reporting capabilities.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-1abc9c?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?logo=spacy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.25-013243?logo=numpy&logoColor=white)

---

🚀 Features

Text Preprocessing – Tokenization, stop-word removal, lemmatization, cleaning

Topic Modeling – LDA (Latent Dirichlet Allocation), NMF (Non-negative Matrix Factorization)

Sentiment Analysis – VADER scoring with sentiment distribution insights

Vectorization – TF-IDF, Bag-of-Words (BoW)

Classification – Supervised learning with cross-validation and optimization

Text Summarization – Extractive summarization using advanced NLP techniques

PDF Export – Four professional report types with customizable styling

Real-time Analysis – Async processing with live status updates

Comprehensive Logging – Detailed logs using Loguru

📂 Project Structure
backend/
├── main.py                  # FastAPI application & API endpoints
├── utils.py                 # Helper functions
├── requirements.txt         # Python dependencies
├── pipeline/                # NLP processing modules
│   ├── preprocessing.py     # Text cleaning & preprocessing
│   ├── topic_modeling.py    # LDA/NMF topic modeling
│   ├── sentiment.py         # Sentiment analysis (VADER)
│   ├── classification.py    # ML classification models
│   ├── summarization.py     # Extractive summarization
│   └── reporting.py         # PDF report generation
├── uploads/                 # User uploaded files
├── results/                 # Cached analysis results (JSON)
├── outputs/                 # Generated charts & PDFs
├── models/                  # Trained ML models
└── logs/                    # Application logs

🌐 API Endpoints
Core Endpoints
Method	Endpoint	Description
GET	/	Root endpoint – backend status
GET	/health	Health check with timestamp
GET	/status	AI pipeline initialization status
Upload & Analysis
Method	Endpoint	Description
POST	/upload	Upload CSV/TXT files
POST	/analyze	Run complete NLP pipeline
POST	/analyze/step	Run single step (debugging)
Results & Reports
Method	Endpoint	Description
GET	/results	List all analysis results
GET	/results/{filename}	Retrieve JSON results
POST	/generate-pdf/{filename}	Generate PDF report
GET	/download/report/{type}/{filename}	Download PDF
Report Types

executive_summary → Business overview with key insights

detailed_analysis → Comprehensive technical analysis

visual_report → Charts and visual representation

overall_report → Complete combined analysis

🛠 Setup & Installation
Prerequisites

Python 3.8+

pip package manager

Install Dependencies
cd backend
pip install -r requirements.txt

Download SpaCy Model
python -m spacy download en_core_web_sm

Start the Server
uvicorn main:app --reload --host 127.0.0.1 --port 8001


Server URL: http://localhost:8001

API Docs: http://localhost:8001/docs (Swagger UI)

📦 Key Dependencies
# FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.2

# Data Processing & ML
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2

# NLP
nltk==3.8.1
spacy==3.7.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# PDF Generation
fpdf2==2.7.6
python-docx==1.1.0
PyPDF2==3.0.1

# Logging
loguru==0.7.2

🔬 Analysis Pipeline

Data Preprocessing

Cleaning, normalization, tokenization, lemmatization

Stop-word removal & feature engineering

Vectorization

TF-IDF and Bag-of-Words

Automatic best model selection based on coherence

Topic Modeling

LDA with BoW

NMF with TF-IDF

Coherence scoring and model selection

Sentiment Analysis

VADER scoring

Positive/Neutral/Negative distribution

Sentiment insights

Classification

Supervised learning models

Cross-validation and metrics: Accuracy, Precision, Recall, F1

Summarization

Extractive sentence selection

Summary quality evaluation

Report Generation

Professional PDFs with charts & tables

Four distinct report types with custom color schemes

📊 Example Usage
Upload File
curl -X POST "http://localhost:8001/upload" -F "file=@dataset.csv"

Run Analysis
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"filename": "dataset.csv", "text_column": "text"}'

Download Report
curl -O "http://localhost:8001/download/report/overall_report/dataset.csv"

🎨 PDF Report Styling

Header Background: Dark Navy #0B1220

Main Title: Turquoise #00C4A7

Section Subtitle: Amber #FFC107

Section Headers: Blue #3A73F0

Subsection Headers: Green #00B894

Body Text: Dark Gray #222222

🔍 Logging

Timestamped entries

Color-coded log levels

Detailed error tracing

Logs stored in logs/

🚀 Production Deployment
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4


Or using Gunicorn:

gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

🤝 Frontend Integration

CORS enabled for local dev

JSON response format

File upload support

Async processing & status updates

Automatic caching of results

📝 Notes

Uploads: uploads/

Analysis cache: results/

Generated PDFs: outputs/

Trained models: models/

Rotating logs: logs/

Built with FastAPI, SpaCy, scikit-learn, and modern NLP techniques 🚀