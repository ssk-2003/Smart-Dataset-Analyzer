📊 Smart Dataset Analyzer

Smart Dataset Analyzer is a full-stack web application for advanced, interactive text dataset analysis and automated report generation using Machine Learning (ML) and Natural Language Processing (NLP).

Designed for both technical and non-technical users, it allows you to:
✅ Upload text datasets
✅ Explore themes, sentiment & topics
✅ Visualize insights interactively
✅ Export professional PDF reports instantly

🚀 Features
🔹 Frontend (React + Tailwind CSS)

Modern Single Page Application (SPA) with responsive UI

File upload dashboard with smooth animations

Interactive visualizations (charts, tables, word clouds, insights)

One-click PDF report download

🔹 Backend (FastAPI + ML/NLP)

Text Preprocessing: tokenization, stop-word removal, lemmatization

Topic Modeling: LDA (Latent Dirichlet Allocation) & NMF (Non-negative Matrix Factorization)

Sentiment Analysis: powered by TextBlob

Vectorization: TF-IDF & Bag-of-Words

Automated Insights & Recommendations

PDF Report Export with FPDF

🔹 Visualizations

📌 Sentiment Bar Charts
📌 Topic Distribution Tables
📌 Word Clouds of top terms
📌 Insights & Recommendations Panel

🔹 File Support

.txt

.csv

.docx

📂 Project Structure
🖥 Backend (backend/)

analyze_all.py → Main FastAPI application

Endpoints:

/analyze-all → POST endpoint for file upload & full analysis

/download-report → POST endpoint for PDF export

requirements.txt → Backend dependencies

Run Backend

cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn analyze_all:app --reload --host 127.0.0.1 --port 8001

🌐 Frontend (frontend/)

Built with React + Tailwind CSS

components/ → Reusable UI (charts, tables, cards, word clouds)

pages/ → Upload page, Analysis dashboard, Landing page

App.js → Router + Page structure

Run Frontend

cd frontend
npm install
npm start

🔗 Connecting Frontend & Backend

Frontend communicates with Backend:

http://localhost:8001/analyze-all → Upload & Analyze

http://localhost:8001/download-report → Export PDF

🗂 Component Overview
📌 Frontend Components

UploadPage.jsx → File upload UI

AnalysisPage.jsx → Dashboard (results, charts, insights, PDF export)

SentimentBarChart.jsx → Sentiment visualization

TopicTable.jsx → Topic keywords & percentages

WordCloudPanel.jsx → Word cloud of top extracted terms

InsightsCard.jsx → Key insights & recommendations

ExportReportButton.jsx → Download report

📌 Backend (analyze_all.py)

Preprocessing: tokenization, lemmatization, stop-word removal

Analysis: topic modeling, sentiment, tf-idf

PDF Export: auto-generated professional report

📄 Requirements
🔹 Backend

Python 3.8+

fastapi, uvicorn, scikit-learn, textblob, spacy, pandas, numpy, python-docx, fpdf

Example: backend/requirements.txt

fastapi
uvicorn
pandas
numpy
scikit-learn
textblob
spacy
python-docx
fpdf

🔹 Frontend

Node.js & npm

React 18+

Tailwind CSS

Recharts (charts)

Framer Motion (animations)

Example: frontend/package.json

{
  "dependencies": {
    "react": "^18.x",
    "framer-motion": "^7.x",
    "recharts": "^2.x",
    "tailwindcss": "^3.x"
  }
}

💡 Usage

1️⃣ Start backend server
2️⃣ Start frontend dev server
3️⃣ Open browser → Upload text file via /upload
4️⃣ Instantly view analysis & visualizations
5️⃣ Click PDF button → Download full report

🎨 Customization

Add your logo, theme, landing page in React

Deploy Frontend → Vercel / Netlify

Deploy Backend → Render / Azure / AWS

Update CORS origins for production

🏁 Example Workflow

✔ Upload .csv dataset
✔ Get topics, sentiments, key terms
✔ Explore visualizations in dashboard
✔ Export PDF summary report

🔥 With Smart Dataset Analyzer, you can turn raw text into actionable insights in minutes!