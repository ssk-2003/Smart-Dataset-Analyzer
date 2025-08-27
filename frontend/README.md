# 🌐 Frontend – Smart Dataset Analyzer  

The **frontend** is a modern **React + Tailwind CSS SPA** providing an **interactive dashboard** for text analysis results.  

---

## 🚀 Features  
- Responsive **file upload dashboard**  
- Interactive **charts, tables, word clouds**  
- Smooth animations (**Framer Motion**)  
- One-click **PDF report download**  

---

## 📂 Structure  
frontend/
│── src/
│ ├── components/ # Reusable UI components (charts, cards, wordclouds)
│ ├── pages/ # Page-level views (Upload, Analysis, Landing)
│ └── App.js # Router + layout
│── package.json
│── README.md

yaml
Copy code

---

## 🛠 Setup & Run  
```bash
cd frontend
npm install
npm start
Runs at → http://localhost:3000/

🔗 API Endpoints Used
http://localhost:8001/analyze-all → Upload & analyze dataset

http://localhost:8001/download-report → Export PDF

📌 Components Overview
UploadPage.jsx → File upload UI

AnalysisPage.jsx → Dashboard (charts, results, export button)

SentimentBarChart.jsx → Sentiment visualization

TopicTable.jsx → Topic keywords & percentages

WordCloudPanel.jsx → Word cloud of top extracted terms

InsightsCard.jsx → Key insights & recommendations

ExportReportButton.jsx → PDF download