# 💻 Smart NLP Analytics Dashboard – Frontend

The Smart NLP Analytics Dashboard frontend is built with React 18, TypeScript, and Vite, delivering a modern, responsive interface for text analysis with interactive visualizations and PDF reporting.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?logo=tailwind-css&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-2.15-8884d8?logo=chartdotjs&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-latest-000000?logo=shadcnui&logoColor=white)

---

## 📸 Application Screenshots

### 🏠 Landing Page
<div align="center">
  <img src="./assets/landing-page.png" alt="Landing Page" width="80%">
  <p><i>Modern hero section with gradients and feature highlights</i></p>
</div>

### 📤 Upload Page
<div align="center">
  <img src="./assets/upload-page.png" alt="Upload Page" width="80%">
  <p><i>Drag-and-drop file upload with validation and preview</i></p>
</div>

### 📊 Analysis Dashboard
<div align="center">
  <img src="./assets/analysis-page.png" alt="Analysis Dashboard" width="80%">
  <p><i>Real-time analysis with interactive charts and visualizations</i></p>
</div>

---

## 🚀 Features

Modern UI/UX – Clean, professional design with Tailwind CSS

Real-time Analysis – Live updates during NLP processing

Interactive Charts – Dynamic visualizations with Recharts

PDF Export – One-click comprehensive report download

Responsive Design – Desktop and mobile-friendly

Component Library – shadcn/ui for polished UI components

Type Safety – Full TypeScript implementation

State Management – React Context API for global state

File Upload – Drag-and-drop CSV/TXT support

Dark Theme – Beautiful gradients and color schemes

---

## 📂 Project Structure

```
frontend/
├── components/                 # React UI components
│   ├── AnalysisPage.tsx        # Main analysis dashboard
│   ├── HeroPage.tsx            # Landing page
│   ├── UploadPage.tsx          # File upload interface
│   └── ui/                     # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── progress.tsx
│       ├── tabs.tsx
│       └── ... (40+ components)
├── contexts/                   # Global state management
│   └── AnalysisContext.tsx
├── hooks/                      # Custom React hooks
│   └── use-toast.ts
├── lib/                        # Utilities
│   └── utils.ts
├── pages/                      # Page components
│   └── Index.tsx
├── App.tsx                     # Root component
├── main.tsx                    # App bootstrap
├── App.css                     # Global styles
├── index.css                   # Tailwind CSS imports
└── README.md                   # Frontend documentation
```

---

## 🎨 Tech Stack

### Core

React 18 → Modern UI library

TypeScript 5.8 → Type-safe development

Vite 5.4 → Lightning-fast build tool

### Styling & UI

Tailwind CSS → Utility-first CSS framework

shadcn/ui → High-quality component library

Radix UI → Accessible primitives

Lucide React → Icon library

Framer Motion → Smooth animations

### Data Visualization

Recharts → Responsive chart library

Custom visualizations → Sentiment, topics, ROC curves

### State & Forms

React Context API → Global state

React Hook Form → Form handling

Zod → Schema validation

### Routing

React Router DOM → Client-side routing

---

## 🛠 Setup & Installation

### Prerequisites

Node.js 18+ and npm

Git

### Install Dependencies

```bash
npm install
```

### Start Development Server

```bash
npm run dev
```


Runs at → http://localhost:5173

### Build for Production

```bash
npm run build
```


Output → dist/ folder ready for deployment

---

## 📦 Key Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.30.1",
    "tailwindcss": "^3.4.17",
    "recharts": "^2.15.4",
    "lucide-react": "^0.462.0",
    "framer-motion": "^12.23.16",
    "@radix-ui/react-*": "^1.x.x",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.6.0",
    "zod": "^3.25.76"
  },
  "devDependencies": {
    "vite": "^5.4.19",
    "typescript": "^5.8.3",
    "@vitejs/plugin-react-swc": "^3.11.0"
  }
}
```

---

## 🎯 Key Features

### Landing Page

Hero section with gradients

Feature highlights

Quick-start call-to-action

Fully responsive

### Upload Page

Drag-and-drop file upload

CSV/TXT support

File validation & preview

Column selection for text analysis

### Analysis Dashboard

Preprocessing Section: Data cleaning stats, vectorization, algorithm selection

Topic Modeling: LDA vs NMF comparison, top words per topic, visual distribution

Sentiment Analysis: Donut charts with positive/neutral/negative breakdown

Visualization Section: ROC curves, confusion matrices, classification metrics

Export Report: One-click PDF download with professional formatting

---

## 🎨 Color Scheme

### Sentiment Charts
Positive: #00D4AA
Neutral:  #FFB800
Negative: #FF6B6B

### Chart Backgrounds
Sentiment: slate-900 → purple-900
Topic Modeling: emerald-600 → cyan-700
ROC Curve: indigo-600 → pink-700
Confusion Matrix: orange-600 → pink-700


Gradients, hover effects, shadow styling included

---

## 🔗 API Integration

Connects to backend at http://localhost:8001:

```typescript
const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8001/upload', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};

const analyzeFile = async (filename: string, textColumn: string) => {
  const response = await fetch('http://localhost:8001/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, text_column: textColumn })
  });
  
  return response.json();
};
```

---

## 📊 Component Highlights

### Interactive Charts (Recharts)

```tsx
<PieChart>
  <Pie data={sentimentData} innerRadius={30} outerRadius={90} dataKey="value">
    <Cell fill="url(#positiveGradient)" />
    <Cell fill="url(#neutralGradient)" />
    <Cell fill="url(#negativeGradient)" />
  </Pie>
</PieChart>
```

### shadcn/ui Components

```tsx
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
```

---

## 🚀 Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

---

## 🎯 State Management

### AnalysisContext.tsx

Holds global state: analysis results, file upload status, current step, error handling, PDF generation

```typescript
import { useAnalysis } from '@/contexts/AnalysisContext';

const { analysisResults, uploadFile, analyzeFile, downloadReport } = useAnalysis();
```

---

## 📱 Responsive Design

Mobile-first approach, touch-friendly layout

Tablet optimized grids

Full dashboard experience on desktop

### Tailwind Breakpoints

sm: 640px
md: 768px
lg: 1024px
xl: 1280px
2xl: 1536px

---

## 🔍 Best Practices

Modular & reusable components

Full TypeScript coverage

Code splitting & lazy loading

Accessibility (ARIA labels, keyboard navigation)

Graceful error handling & loading states

---

## 🌐 Deployment

### Build for production:

```bash
npm run build
```


Deploy static files (dist/) to Vercel, Netlify, or GitHub Pages

### .env setup:

```bash
VITE_API_URL=http://localhost:8001
```

---

## 🤝 Backend Integration

Real-time updates from FastAPI backend

File upload with progress tracking

PDF report download

Error handling & toast notifications

Built with React, TypeScript, Tailwind CSS, and modern web technologies 🚀