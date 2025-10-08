# 💻 Smart NLP Analytics Dashboard – Frontend

The Smart NLP Analytics Dashboard frontend is built with **React 18**, **TypeScript**, and **Vite**, delivering a modern, responsive interface for text analysis with interactive visualizations and PDF reporting.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?logo=tailwind-css&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-2.15-8884d8?logo=chartdotjs&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-latest-000000?logo=shadcnui&logoColor=white)

---

## 🚀 Features

{{ ... }}
✅ **Modern UI/UX** → Clean, professional design with Tailwind CSS  
✅ **Real-time Analysis** → Live updates during NLP processing  
✅ **Interactive Charts** → Dynamic visualizations using Recharts  
✅ **PDF Export** → One-click comprehensive report download  
✅ **Responsive Design** → Works seamlessly on desktop and mobile  
✅ **Component Library** → shadcn/ui for polished UI components  
✅ **Type Safety** → Full TypeScript implementation  
✅ **State Management** → React Context API for global state  
✅ **File Upload** → Drag-and-drop CSV/TXT file support  
✅ **Dark Theme** → Beautiful color schemes with gradients  

---

## 📂 Project Structure

```
src/
├── components/                 # React components
│   ├── AnalysisPage.tsx        # Main analysis dashboard
│   ├── HeroPage.tsx            # Landing page
│   ├── UploadPage.tsx          # File upload interface
│   └── ui/                     # shadcn/ui component library
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── progress.tsx
│       ├── tabs.tsx
│       └── ... (40+ components)
├── contexts/                   # State management
│   └── AnalysisContext.tsx     # Global analysis state
├── hooks/                      # Custom React hooks
│   └── use-toast.ts            # Toast notifications
├── lib/                        # Utilities
│   └── utils.ts                # Helper functions
├── pages/                      # Page components
│   └── Index.tsx               # Main app entry
├── App.tsx                     # Root component
└── main.tsx                    # Application bootstrap
```

---

## 🎨 Tech Stack

### **Core Framework**
- **React 18.3** → Modern UI library
- **TypeScript 5.8** → Type-safe development
- **Vite 5.4** → Lightning-fast build tool

### **Styling & UI**
- **Tailwind CSS** → Utility-first CSS framework
- **shadcn/ui** → High-quality component library
- **Radix UI** → Accessible primitives
- **Lucide React** → Beautiful icon library
- **Framer Motion** → Smooth animations

### **Data Visualization**
- **Recharts** → Responsive chart library
- **Custom Charts** → Sentiment, topic modeling, ROC curves

### **State & Forms**
- **React Context API** → Global state management
- **React Hook Form** → Form handling
- **Zod** → Schema validation

### **Routing & Navigation**
- **React Router DOM** → Client-side routing

---

## 🛠 Setup & Installation

### **Prerequisites**
- Node.js 18+ and npm
- Git

### **1. Install Dependencies**

```bash
npm install
```

### **2. Start Development Server**

```bash
npm run dev
```

**Frontend runs at** → `http://localhost:5173`

### **3. Build for Production**

```bash
npm run build
```

**Output** → `dist/` folder ready for deployment

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
    "class-variance-authority": "^0.7.1",
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

### **1. Landing Page (HeroPage.tsx)**
- Modern hero section with gradients
- Feature highlights
- Quick start CTA buttons
- Responsive design

### **2. Upload Page (UploadPage.tsx)**
- Drag-and-drop file upload
- CSV/TXT file support
- File validation
- Column selection for text data
- Preview uploaded data

### **3. Analysis Dashboard (AnalysisPage.tsx)**
- **Preprocessing Section**
  - Data cleaning statistics
  - Vectorization methods (TF-IDF, BoW)
  - Algorithm selection display

- **Topic Modeling Section**
  - LDA vs NMF comparison
  - Coherence scores
  - Top words per topic
  - Visual topic distribution

- **Sentiment Analysis Section**
  - Donut chart with gradients
  - Positive/Neutral/Negative breakdown
  - Descriptive labels (Optimistic/Balanced/Critical)
  - Distribution statistics

- **Visualization Section**
  - ROC curves with vibrant gradients
  - Confusion matrices
  - Classification metrics cards
  - Performance statistics

- **Export Report**
  - One-click PDF download
  - Comprehensive report generation
  - Professional formatting

---

## 🎨 Color Scheme

### **Sentiment Chart Colors**
```css
Positive: Bright teal/cyan (#00D4AA)
Neutral: Bright amber/gold (#FFB800)
Negative: Coral red (#FF6B6B)
```

### **Chart Backgrounds**
```css
Sentiment: Dark slate with purple (slate-900 → purple-900)
Topic Modeling: Emerald to cyan (emerald-600 → cyan-700)
ROC Curve: Indigo to pink (indigo-600 → pink-700)
Confusion Matrix: Orange to pink (orange-600 → pink-700)
```

### **Gradients**
- Smooth transitions with `via-` colors
- Hover effects with scale transforms
- Shadow effects matching themes

---

## 🔗 API Integration

The frontend connects to the backend at `http://localhost:8001`:

```typescript
// Example API calls from AnalysisContext.tsx
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

### **Interactive Charts (Recharts)**
```tsx
// Donut chart for sentiment
<PieChart>
  <Pie
    data={sentimentData}
    innerRadius={30}
    outerRadius={90}
    dataKey="value"
  >
    <Cell fill="url(#positiveGradient)" />
    <Cell fill="url(#neutralGradient)" />
    <Cell fill="url(#negativeGradient)" />
  </Pie>
</PieChart>
```

### **shadcn/ui Components**
```tsx
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
```

---

## 🚀 Available Scripts

```bash
npm run dev          # Start development server (Vite)
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

---

## 🎯 State Management

### **AnalysisContext.tsx**
Provides global state for:
- Analysis results
- File upload status
- Current analysis step
- Error handling
- PDF report generation

```tsx
import { useAnalysis } from '@/contexts/AnalysisContext';

const { analysisResults, uploadFile, analyzeFile, downloadReport } = useAnalysis();
```

---

## 📱 Responsive Design

The interface adapts to all screen sizes:
- **Mobile**: Stack layout, touch-friendly
- **Tablet**: Optimized grid layouts
- **Desktop**: Full dashboard experience

Tailwind breakpoints:
```css
sm: 640px   /* Small devices */
md: 768px   /* Medium devices */
lg: 1024px  /* Large devices */
xl: 1280px  /* Extra large devices */
2xl: 1536px /* 2X Extra large devices */
```

---

## 🔍 Best Practices

✅ **Component Organization** → Modular, reusable components  
✅ **Type Safety** → Full TypeScript coverage  
✅ **Performance** → Code splitting, lazy loading  
✅ **Accessibility** → ARIA labels, keyboard navigation  
✅ **Error Handling** → Graceful error states  
✅ **Loading States** → Progress indicators  
✅ **Responsive** → Mobile-first approach  

---

## 🌐 Deployment

### **Build for Production**
```bash
npm run build
```

### **Deploy to Vercel/Netlify**
The `dist/` folder is ready for static hosting:
- Vercel: `vercel deploy`
- Netlify: `netlify deploy --prod`
- GitHub Pages: Push `dist/` to `gh-pages` branch

### **Environment Variables**
Create `.env` file:
```bash
VITE_API_URL=http://localhost:8001
```

---

## 🤝 Integration with Backend

The frontend seamlessly works with the FastAPI backend:
- Real-time analysis updates
- File upload with progress tracking
- PDF report download
- Error handling with user-friendly messages
- Toast notifications for user feedback

---

**Built with React, TypeScript, Tailwind CSS, and modern web technologies** 🚀
