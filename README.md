🌌 Smart DataSet Analyzer – Advanced NLP Text Analysis Platform

Smart DataSet Analyzer is a comprehensive web application for advanced Natural Language Processing (NLP) analysis, including sentiment analysis, topic modeling, text summarization, and professional PDF report generation. It offers both a modern frontend dashboard and a high-performance backend API for end-to-end text analytics.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-1abc9c?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-blue?logo=react&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-blue?logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

---

## 🖼️ UI Preview

<div align="center">
  <img src="./assets/ui-preview.png" alt="Smart DataSet Analyzer - UI Preview" width="100%">
  <p><i>Landing Page • Upload Interface • Analysis Dashboard with Interactive Visualizations</i></p>
</div>

---

🚀 Key Features

Sentiment Analysis – Multi-model detection with VADER and ensemble methods

Topic Modeling – LDA and NMF with coherence scoring

Text Summarization – Extractive summarization with advanced NLP techniques

Classification – Supervised learning with cross-validation and optimization

Professional Reports – Four distinct PDF report types with customizable formatting

Interactive Dashboard – Real-time analysis with beautiful visualizations

Responsive Design – Works seamlessly on desktop and mobile

Error Handling & Logging – Robust error management and detailed Loguru logs

📊 Report Types

Executive Summary – Business-focused overview with key insights

Detailed Analysis – Comprehensive technical analysis with statistics

Visual Report – Chart-centric presentation with interactive visualizations

Overall Report – Complete analysis combining all sections and appendices

🛠 Tech Stack
Frontend

React 18 with TypeScript

Vite for fast builds and development

Tailwind CSS for responsive styling

Recharts for interactive charts

shadcn/ui for polished UI components

Backend

FastAPI for high-performance REST API

Python 3.11+ with advanced NLP libraries

spaCy for natural language processing

scikit-learn for machine learning models

FPDF / python-docx for professional PDF generation

Loguru for logging and debugging

📁 Project Structure
smart-dataset-analyzer/
├── src/                          # Frontend React application
│   ├── components/               # React components
│   │   ├── AnalysisPage.tsx      # Main analysis interface
│   │   ├── HeroPage.tsx          # Landing page
│   │   ├── UploadPage.tsx        # File upload page
│   │   └── ui/                   # UI components (shadcn/ui)
│   ├── contexts/                 # React context providers
│   │   └── AnalysisContext.tsx   # Global state management
│   ├── hooks/                    # Custom React hooks
│   ├── lib/                      # Utility functions
│   ├── pages/                    # Pages
│   └── README.md                 # Frontend documentation
├── backend/                      # Python FastAPI backend
│   ├── pipeline/                 # NLP processing modules
│   │   ├── classification.py
│   │   ├── preprocessing.py
│   │   ├── reporting.py
│   │   ├── sentiment.py
│   │   ├── summarization.py
│   │   └── topic_modeling.py
│   ├── main.py                   # FastAPI server
│   ├── utils.py                  # Backend utilities
│   └── README.md                 # Backend documentation
├── outputs/                      # Generated charts and PDFs
├── results/                      # Analysis results cache
├── uploads/                      # User uploaded files
└── models/                       # Trained ML models


📚 See backend/README.md
 for API reference and src/README.md
 for frontend setup instructions.

🚀 Getting Started
Prerequisites

Node.js 18+ and npm

Python 3.11+

Git

Installation

Clone Repository

git clone <repository-url>
cd smart-dataset-analyzer


Setup Frontend

cd src
npm install


Setup Backend

cd backend

# Create virtual environment
python -m venv smart-env

# Activate virtual environment
# Windows:
smart-env\Scripts\activate
# macOS/Linux:
source smart-env/bin/activate

# Install dependencies
pip install -r requirements.txt

Running the Application

Start Backend Server

cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8001


Runs at: http://localhost:8001

Start Frontend Development Server

cd src
npm run dev


Runs at: http://localhost:5173

📖 Usage

Upload Data – CSV/TXT files with text content

Run Analysis – Process data through NLP pipeline

View Results – Explore interactive dashboards: sentiment, topics, and metrics

Generate Reports – Export professional PDFs

Export Data – Save results and charts

🔧 API Endpoints

POST /upload – Upload files for analysis

POST /analyze – Run complete NLP analysis

POST /generate-pdf/{filename} – Generate PDF reports

GET /download/report/{type}/{filename} – Download generated reports

📈 Analysis Pipeline

Data Preprocessing – Cleaning, normalization, tokenization

Feature Engineering – TF-IDF & Bag-of-Words

Topic Modeling – LDA & NMF with coherence scoring

Sentiment Analysis – VADER & ensemble methods

Classification – Supervised learning with cross-validation

Summarization – Extractive summarization

Report Generation – Professional PDFs with charts and tables

🎨 Visualization & Design

Real-time Processing – Live updates during analysis

Interactive Charts – Sentiment, topics, ROC curves, and confusion matrices

Responsive UI – Desktop, tablet, and mobile-friendly

Professional Reports – Export-ready PDF documents

🤝 Contributing

Fork the repository

Create a feature branch: git checkout -b feature/my-feature

Commit changes: git commit -m "Add new feature"

Push branch: git push origin feature/my-feature

Open a Pull Request

📄 License

MIT License – see LICENSE file for details

🙏 Acknowledgments

Built using modern web technologies and advanced NLP libraries

Inspired by the need for professional, accessible text analysis tools

Thanks to the open-source community for excellent libraries

Smart DataSet Analyzer – Transforming text data into actionable insights with professional-grade analysis and reporting. 🚀