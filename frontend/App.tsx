import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState } from "react";
import HeroPage from "./components/HeroPage";
import UploadPage from "./components/UploadPage";
import AnalysisPage from '@/components/AnalysisPage';
import { AnalysisProvider } from "./contexts/AnalysisContext";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => {
  const [currentPage, setCurrentPage] = useState<'hero' | 'upload' | 'analysis'>('hero'); // Default to hero page
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [serverFilename, setServerFilename] = useState<string>('');

  const handleGetStarted = () => {
    setCurrentPage('upload');
  };

  const handleBack = () => {
    setCurrentPage('upload');
  };

  const handleBackToHero = () => {
    setCurrentPage('hero');
  };

  const handleUploadComplete = (file: File, filename: string) => {
    setUploadedFile(file);
    setServerFilename(filename);
    setCurrentPage('analysis');
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'analysis':
        return (
          <AnalysisProvider>
            <AnalysisPage
              onBack={handleBack}
              uploadedFile={uploadedFile}
              serverFilename={serverFilename}
            />
          </AnalysisProvider>
        );
      case 'upload':
        return (
          <UploadPage
            onBack={handleBackToHero}
            onUploadComplete={handleUploadComplete}
          />
        );
      default:
        return <HeroPage onGetStarted={handleGetStarted} />;
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={renderCurrentPage()} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;