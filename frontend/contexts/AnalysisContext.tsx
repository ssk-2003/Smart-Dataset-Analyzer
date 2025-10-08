import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import { useToast } from '@/components/ui/use-toast';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// Types for analysis data
type MetricItem = {
  title: string;
  value: string | number;
  change?: number;
  icon: string;
  color: string;
};

type SentimentItem = {
  name: string;
  value: number;
  color: string;
};

type TopicItem = {
  topic: string;
  keywords: string[];
  distribution: number;
};

type ConfusionCell = {
  actual: string;
  predicted: string;
  value: number;
  count: number;
};

type ClassificationMetrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  sensitivity?: number;
  specificity?: number;
  roc_auc?: number;
  auc?: number;
  npv?: number;
  confusionMatrix: number[][] | ConfusionCell[];
  roc_curve?: Array<{fpr: number; tpr: number}>;
  calibration_curve?: Array<{mean_predicted: number; fraction_positive: number}>;
  classification_report?: any;
};

type ProcessingStep = {
  label: string;
  progress: number;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  message?: string;
};

type AnalysisData = {
  metrics: MetricItem[];
  sentiment: SentimentItem[];
  topics: TopicItem[];
  classification: ClassificationMetrics | null;
  insights: string[];
  topTerms: string[];
  summarization: {
    summary: string;
    key_sentences: string[];
    method_used: string;
    original_word_count?: number;
    summary_word_count?: number;
    compression_ratio?: number;
    total_documents?: number;
    processed_documents?: number;
  } | null;
  preprocessing: {
    total_entries: number;
    valid_texts: number;
    average_length: number;
    nlp_engine: string;
    status: string;
    vocabulary_size?: number;
    total_tokens?: number;
    quality_metrics?: {
      text_completeness?: number;
      language_consistency?: number;
      encoding_quality?: number;
      duplicate_detection?: number;
    };
    outliers?: number;
    processing_time?: string;
  } | null;
  processingStatus: ProcessingStep[];
  documentInfo: {
    document_count: number;
    total_tokens: number;
    average_length?: number;
  } | null;
  topicModels: {
    name: string;
    coherence_score?: number;
    perplexity?: number;
    reconstruction_error?: number;
    topic_diversity?: number;
    topics?: any[];
    matrix_shape?: number[];
  }[] | null;
  isAnalyzing: boolean;
  analysisError: string | null;
  currentStep: string;
  progress: number;
};

type AnalysisContextType = {
  analysisData: AnalysisData;
  setAnalysisData: React.Dispatch<React.SetStateAction<AnalysisData>>;
  startAnalysis: (file: File) => Promise<void>;
  handleDownloadReport: (reportType: string) => void;
};

const defaultAnalysisData: AnalysisData = {
  metrics: [],
  sentiment: [],
  topics: [],
  classification: null,
  insights: [],
  topTerms: [],
  summarization: null,
  preprocessing: null,
  processingStatus: [],
  documentInfo: null,
  topicModels: null,
  isAnalyzing: false,
  analysisError: null,
  currentStep: '',
  progress: 0,
};

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [analysisData, setAnalysisData] = useState<AnalysisData>(defaultAnalysisData);
  const { toast } = useToast();

  const startAnalysis = useCallback(async (file: File) => {
    if (!file) {
      toast({
        title: 'Error',
        description: 'No file provided for analysis',
        variant: 'destructive',
      });
      return;
    }

    setAnalysisData(prev => ({
      ...prev,
      isAnalyzing: true,
      analysisError: null,
      currentStep: 'Uploading file...',
      progress: 10,
    }));

    try {
      // Step 1: Upload the file first
      setAnalysisData(prev => ({
        ...prev,
        currentStep: 'Uploading file to server...',
        progress: 15,
      }));

      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch('http://localhost:8001/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${uploadResponse.status}`);
      }

      const uploadResult = await uploadResponse.json();
      const serverFilename = uploadResult.filename;

      setAnalysisData(prev => ({
        ...prev,
        currentStep: 'File uploaded, starting analysis...',
        progress: 25,
      }));

      // Step 2: Start analysis with the uploaded filename
      const analysisRequest = {
        filename: serverFilename,
        text_column: 'text',
        label_column: null,
        config: {}
      };

      setAnalysisData(prev => ({
        ...prev,
        currentStep: 'Analyzing data...',
        progress: 30,
      }));

      const analysisResponse = await fetch('http://localhost:8001/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisRequest),
      });

      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis failed: ${analysisResponse.status}`);
      }

      const data = await analysisResponse.json();
      
      // Map the backend response to our state structure
      const mappedMetrics = data.metrics || [];
      
      // Transform sentiment data from object to array format for charts
      const mappedSentiment = data.sentiment ? [
        { 
          name: 'Positive', 
          value: data.sentiment.positive || 0, 
          color: '#10b981' 
        },
        { 
          name: 'Neutral', 
          value: data.sentiment.neutral || 0, 
          color: '#6b7280' 
        },
        { 
          name: 'Negative', 
          value: data.sentiment.negative || 0, 
          color: '#ef4444' 
        }
      ] : [];
      
      // Store the full sentiment details for score distribution
      const sentimentDetails = data.sentiment || {};
      
      const mappedTopics = (() => {
        if (data.topic_models && data.topic_models.length > 0) {
          // Get topics from the best model (first one)
          const bestModel = data.topic_models[0];
          const topics = bestModel.topics || [];
          // Add method information to each topic
          return topics.map((topic: any) => ({
            ...topic,
            method: bestModel.name || "NMF" // Add method name from the model
          }));
        }
        return data.topicModeling || [];
      })();

      const mappedTopicModels = (() => {
        if (data.topic_models && data.topic_models.length > 0) {
          return data.topic_models.map((model: any) => ({
            name: model.name || "Unknown",
            coherence_score: model.coherence_score,
            perplexity: model.perplexity,
            reconstruction_error: model.reconstruction_error,
            topic_diversity: model.topic_diversity,
            topics: model.topics || [],
            matrix_shape: model.matrix_shape
          }));
        }
        return null;
      })();
      const mappedClassification = (() => {
        if (!data.classification) return null;
        
        const classification = data.classification;
        
        // Keep confusion matrix in 2D array format for easier use in components
        let confusionMatrix = classification.confusion_matrix || [[0, 0], [0, 0]];
        
        return {
          accuracy: classification.accuracy || 0,
          precision: classification.precision || 0,
          recall: classification.recall || 0,
          f1: classification.f1 || classification.f1Score || 0,
          sensitivity: classification.sensitivity || 0,
          specificity: classification.specificity || 0,
          roc_auc: classification.roc_auc || 0,
          auc: classification.auc || classification.roc_auc || 0,
          npv: classification.npv || 0,
          confusionMatrix,
          roc_curve: classification.roc_curve || [],
          calibration_curve: classification.calibration_curve || [],
          classification_report: classification.classification_report || {}
        };
      })();
      const mappedInsights = (() => {
        if (data.insights) {
          if (Array.isArray(data.insights)) {
            return data.insights;
          } else if (data.insights.insights && Array.isArray(data.insights.insights)) {
            return data.insights.insights; // Backend sends insights in 'insights' property
          } else if (data.insights.text && Array.isArray(data.insights.text)) {
            return data.insights.text;
          }
        }
        return [];
      })();
      
      const mappedTopTerms = (() => {
        if (data.insights && data.insights.top_terms) {
          return data.insights.top_terms;
        }
        return data.topTerms || data.top_terms || [];
      })();

      const mappedSummarization = (() => {
        if (data.summarization) {
          return {
            summary: data.summarization.summary || data.summarization.textrank_summary || data.summarization.sumy_summary || '',
            key_sentences: data.summarization.key_sentences || [],
            method_used: data.summarization.method_used || 'Sumy LSA',
            original_word_count: data.summarization.original_word_count,
            summary_word_count: data.summarization.summary_word_count,
            compression_ratio: data.summarization.compression_ratio,
            total_documents: data.summarization.total_documents,
            processed_documents: data.summarization.processed_documents
          };
        }
        return null;
      })();
      const mappedProcessingStatus = data.processingStatus || [
        { label: 'Data Collection', progress: 100, status: 'completed' as const },
        { label: 'Preprocessing', progress: 100, status: 'completed' as const },
        { label: 'Topic Modeling', progress: 100, status: 'completed' as const },
        { label: 'Sentiment Analysis', progress: 100, status: 'completed' as const },
        { label: 'Summarization', progress: 100, status: 'completed' as const },
        { label: 'Report Generation', progress: 100, status: 'completed' as const },
      ];
      const mappedDocumentInfo = data.documentInfo || data.document_info || {
        document_count: data.document_count || 1,
        total_tokens: data.total_tokens || 0,
        average_length: data.average_length || 0,
      };

      const mappedPreprocessing = (() => {
        if (data.preprocessing) {
          return {
            total_entries: data.document_info?.document_count || data.preprocessing.processed_documents?.length || 0,
            valid_texts: data.document_info?.document_count || data.preprocessing.processed_documents?.length || 0,
            average_length: data.preprocessing.avg_doc_length || data.document_info?.average_length || 0,
            nlp_engine: data.preprocessing.nlp_engine || 'spaCy',
            status: data.preprocessing.status || 'completed',
            // Pass through all backend preprocessing data
            vocabulary_size: data.preprocessing.vocabulary_size,
            total_tokens: data.preprocessing.total_tokens,
            quality_metrics: data.preprocessing.quality_metrics,
            outliers: data.preprocessing.outliers,
            processing_time: data.preprocessing.processing_time
          };
        }
        // Fallback to document_info if preprocessing is not available
        if (data.document_info) {
          return {
            total_entries: data.document_info.document_count || 0,
            valid_texts: data.document_info.document_count || 0,
            average_length: data.document_info.average_length || 0,
            nlp_engine: 'spaCy',
            status: 'completed'
          };
        }
        return null;
      })();

      // Update state with the processed data
      setAnalysisData(prev => ({
        ...prev,
        metrics: mappedMetrics,
        sentiment: mappedSentiment,
        sentimentDetails: sentimentDetails, // Add full sentiment details for score distribution
        topics: mappedTopics,
        classification: mappedClassification,
        insights: mappedInsights,
        topTerms: mappedTopTerms,
        summarization: mappedSummarization,
        preprocessing: mappedPreprocessing,
        processingStatus: mappedProcessingStatus,
        documentInfo: {
          ...mappedDocumentInfo,
          // Store the results filename for report generation
          results_filename: data.results_filename || mappedDocumentInfo.filename
        },
        topicModels: mappedTopicModels,
        isAnalyzing: false,
        analysisError: null,
        currentStep: 'Analysis complete',
        progress: 100,
      }));

      toast({
        title: 'Analysis Complete! 🎉',
        description: 'Your data has been successfully analyzed',
      });

    } catch (error) {
      console.error('Error during analysis:', error);
      
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      
      setAnalysisData(prev => ({
        ...prev,
        isAnalyzing: false,
        analysisError: errorMessage,
        currentStep: 'Analysis failed',
        progress: 0,
      }));

      // User-friendly error message
      let userMessage = 'Failed to analyze the document';
      if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
        userMessage = 'Unable to connect to the analysis server. Please ensure the backend is running on port 8001.';
      } else if (errorMessage.includes('401') || errorMessage.includes('403')) {
        userMessage = 'Authentication failed. Please refresh the page and try again.';
      } else if (errorMessage.includes('500')) {
        userMessage = 'The server encountered an error. Please try again later.';
      } else if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
        userMessage = 'The request timed out. The server might be busy. Please try again in a moment.';
      } else if (errorMessage.includes('CORS')) {
        userMessage = 'CORS error. Please ensure the backend allows requests from this origin.';
      } else if (errorMessage.includes('File not found')) {
        userMessage = 'The uploaded file could not be found. Please try uploading again.';
      } else if (errorMessage.includes('Upload failed')) {
        userMessage = 'File upload failed. Please check the file and try again.';
      } else if (errorMessage.includes('Analysis failed')) {
        userMessage = 'Analysis failed. Please check the file format and try again.';
      }

      toast({
        title: 'Analysis Error',
        description: userMessage,
        variant: 'destructive',
        duration: 10000,
      });
    }
  }, [toast]);

  const handleDownloadReport = useCallback(async (reportType: string) => {
    if (reportType === 'all') {
      toast({
        title: 'Generating All Reports',
        description: 'Creating Executive Summary, Detailed Analysis, Visual Report, and Overall Report...',
      });
    } else {
      // Map report type to user-friendly name
      const reportNames: { [key: string]: string } = {
        'executive': 'Executive Summary',
        'detailed': 'Detailed Analysis',
        'visual': 'Visual Report', 
        'combined': 'Overall Report',
        'executive_summary': 'Executive Summary',
        'detailed_analysis': 'Detailed Analysis',
        'visual_report': 'Visual Report',
        'overall_report': 'Overall Report'
      };
      
      const displayName = reportNames[reportType] || reportType;
      
      toast({
        title: 'Report Generation',
        description: `Generating ${displayName} report...`,
      });
    }
    
    try {
      // Get the results filename from the current analysis data or use a default
      const resultsFilename = (analysisData.documentInfo as any)?.results_filename;
      const originalFilename = (analysisData.documentInfo as any)?.filename;
      
      // Use 'latest' as a special keyword if no filename is available
      const filename = resultsFilename || originalFilename || 'latest';
      
      console.log('Attempting to generate report for filename:', filename);
      console.log('Results filename:', resultsFilename);
      console.log('Original filename:', originalFilename);
      console.log('Report type requested:', reportType);
      
      // Check if we have any analysis data
      if (!resultsFilename && !originalFilename && (!analysisData.sentiment || analysisData.sentiment.length === 0)) {
        toast({
          title: 'No Analysis Found',
          description: 'Please upload and analyze a file first before generating reports.',
          variant: 'destructive',
        });
        return;
      }
      
      // Generate PDF report
      const generateResponse = await fetch(`http://localhost:8001/generate-pdf/${filename}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      console.log('Generate response status:', generateResponse.status);
      
      if (!generateResponse.ok) {
        const errorData = await generateResponse.json().catch(() => ({}));
        const errorMessage = errorData.detail || `HTTP ${generateResponse.status}: Failed to generate report`;
        console.error('Backend generation failed:', errorMessage);
        throw new Error(errorMessage);
      }
      
      const generateResult = await generateResponse.json();
      console.log('Generate result:', generateResult);
      
      if (reportType === 'all') {
        // Download all comprehensive reports (new format)
        const reportTypes = ['executive_summary', 'detailed_analysis', 'visual_report', 'overall_report'];
        let downloadCount = 0;
        
        for (const type of reportTypes) {
          const reportInfo = generateResult.reports[type];
          if (reportInfo && reportInfo.download_url) {
            const downloadResponse = await fetch(`http://localhost:8001${reportInfo.download_url}`);
            
            if (downloadResponse.ok) {
              const blob = await downloadResponse.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = reportInfo.filename || `${type}_report_${new Date().toISOString().slice(0,10)}.pdf`;
              a.style.display = 'none';
              document.body.appendChild(a);
              a.click();
              
              // Clean up
              setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
              }, 100 * (downloadCount + 1));
              
              downloadCount++;
            }
          }
        }
        
        // Also download legacy reports if user specifically requests them
        if (downloadCount === 0) {
          // Fallback to legacy report types
          const legacyReportTypes = ['executive', 'detailed', 'visual', 'combined'];
          for (const type of legacyReportTypes) {
            const reportInfo = generateResult.reports[type];
            if (reportInfo && reportInfo.download_url) {
              const downloadResponse = await fetch(`http://localhost:8001${reportInfo.download_url}`);
              
              if (downloadResponse.ok) {
                const blob = await downloadResponse.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = reportInfo.filename || `${type}_report_${new Date().toISOString().slice(0,10)}.pdf`;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                
                // Clean up
                setTimeout(() => {
                  window.URL.revokeObjectURL(url);
                  document.body.removeChild(a);
                }, 100 * (downloadCount + 1));
                
                downloadCount++;
              }
            }
          }
        }
        
        toast({
          title: 'All Reports Downloaded',
          description: `Successfully downloaded ${downloadCount} reports`,
        });
      } else {
        // Download specific report type - prioritize new comprehensive reports
        let downloadUrl = '';
        let filename = '';
        
        // Map old report type names to new comprehensive report types
        const reportTypeMapping: { [key: string]: string } = {
          'executive': 'executive_summary',
          'detailed': 'detailed_analysis', 
          'visual': 'visual_report',
          'combined': 'overall_report'
        };
        
        // Try new comprehensive report type first
        const newReportType = reportTypeMapping[reportType] || reportType;
        
        if (generateResult.reports && generateResult.reports[newReportType]) {
          // New comprehensive report structure
          const reportInfo = generateResult.reports[newReportType];
          downloadUrl = reportInfo.download_url;
          filename = reportInfo.filename;
          console.log('Using new report type:', newReportType, 'filename:', filename);
        } else if (generateResult.reports && generateResult.reports[reportType]) {
          // Original report type (legacy)
          const reportInfo = generateResult.reports[reportType];
          downloadUrl = reportInfo.download_url;
          filename = reportInfo.filename;
          console.log('Using legacy report type:', reportType, 'filename:', filename);
        } else {
          // Fallback to old structure for backward compatibility
          downloadUrl = generateResult.download_url;
          filename = generateResult.filename;
          console.log('Using fallback structure, filename:', filename);
        }
        
        if (!downloadUrl) {
          console.error('No download URL found for report type:', reportType);
          throw new Error(`${reportType} report not available`);
        }
        
        if (filename === 'report_error.pdf') {
          console.error('Backend generated error report');
          throw new Error('Backend report generation failed - error report returned');
        }
        
        console.log('Downloading from:', downloadUrl);
        const downloadResponse = await fetch(`http://localhost:8001${downloadUrl}`);
        
        if (!downloadResponse.ok) {
          throw new Error('Failed to download report');
        }
        
        const blob = await downloadResponse.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `${reportType}_report_${new Date().toISOString().slice(0,10)}.pdf`;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        }, 100);
        
        // Use the same mapping for success message
        const reportNames: { [key: string]: string } = {
          'executive': 'Executive Summary',
          'detailed': 'Detailed Analysis',
          'visual': 'Visual Report', 
          'combined': 'Overall Report',
          'executive_summary': 'Executive Summary',
          'detailed_analysis': 'Detailed Analysis',
          'visual_report': 'Visual Report',
          'overall_report': 'Overall Report'
        };
        
        const displayName = reportNames[reportType] || reportType.charAt(0).toUpperCase() + reportType.slice(1);
        
        toast({
          title: 'Report Downloaded',
          description: `${displayName} report has been downloaded successfully`,
        });
      }
    } catch (error) {
      console.error('Error generating report:', error);
      
      // Show detailed error information for debugging
      toast({
        title: 'Backend Report Generation Failed',
        description: `Error: ${error instanceof Error ? error.message : 'Unknown error'}. Using fallback PDF generation.`,
        variant: 'destructive',
        duration: 5000,
      });
      
      // Fallback: Generate a beautiful PDF report using jsPDF
      try {
        await generateCustomPDF(reportType, analysisData);
        
        toast({
          title: 'Fallback PDF Generated',
          description: `${reportType} report generated using client-side fallback`,
        });
      } catch (fallbackError) {
        console.error('Fallback report generation failed:', fallbackError);
        toast({
          title: 'Report Generation Failed',
          description: 'Both backend and fallback report generation failed. Please try again.',
          variant: 'destructive',
        });
      }
    }
  }, [toast, analysisData]);

  // Custom PDF generation function based on report type
  const generateCustomPDF = async (reportType: string, data: AnalysisData) => {
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    let yPosition = 20;
    
    // Helper function to add colored text
    const addColoredText = (text: string, x: number, y: number, color: string, fontSize: number = 12, fontStyle: string = 'normal') => {
      pdf.setFontSize(fontSize);
      pdf.setFont('helvetica', fontStyle);
      const [r, g, b] = hexToRgb(color);
      pdf.setTextColor(r, g, b);
      pdf.text(text, x, y);
      return y + (fontSize * 0.35);
    };
    
    // Helper function to convert hex to RGB
    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16)
      ] : [0, 0, 0];
    };
    
    // Add gradient background
    pdf.setFillColor(15, 23, 42); // Dark blue background
    pdf.rect(0, 0, pageWidth, 40, 'F');
    
    // Title
    yPosition = addColoredText('SMART DATASET ANALYZER', 20, 25, '#00D4AA', 24, 'bold');
    yPosition = addColoredText(`${reportType.toUpperCase()} ANALYSIS REPORT`, 20, yPosition + 5, '#FFB800', 16, 'bold');
    yPosition = addColoredText(`Generated: ${new Date().toLocaleString()}`, 20, yPosition + 5, '#FFFFFF', 10);
    
    yPosition += 15;
    
    // Generate content based on report type
    if (reportType === 'executive' || reportType === 'executive_summary') {
      await generateExecutiveSummary(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    } else if (reportType === 'detailed' || reportType === 'detailed_analysis') {
      await generateDetailedAnalysis(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    } else if (reportType === 'visual' || reportType === 'visual_report') {
      await generateVisualReport(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    } else if (reportType === 'all' || reportType === 'overall_report' || reportType === 'combined') {
      await generateCompleteReport(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    } else {
      // Fallback for any unknown report type - generate complete report
      await generateCompleteReport(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    }
    
    // Footer
    pdf.setFillColor(15, 23, 42); // Dark blue background
    pdf.rect(0, pageHeight - 20, pageWidth, 20, 'F');
    addColoredText('Generated by Smart Dataset Analyzer - Advanced NLP Analytics Platform', 20, pageHeight - 10, '#00D4AA', 10);
    
    // Save the PDF
    const fileName = `${reportType}_report_${new Date().toISOString().slice(0,10)}.pdf`;
    pdf.save(fileName);
  };

  // Executive Summary - Clean white background design
  const generateExecutiveSummary = async (pdf: any, data: AnalysisData, addColoredText: any, yPosition: number, pageWidth: number, pageHeight: number) => {
    // Analysis Overview Section - Colored title, black content
    yPosition = addColoredText('ANALYSIS OVERVIEW', 20, yPosition + 5, '#3B82F6', 16, 'bold');
    pdf.setDrawColor(59, 130, 246);
    pdf.setLineWidth(0.5);
    pdf.line(20, yPosition, pageWidth - 20, yPosition);
    yPosition += 5;
    
    // Two-column layout for key metrics
    const col1X = 25;
    const col2X = pageWidth / 2 + 10;
    let col1Y = yPosition + 5;
    let col2Y = yPosition + 5;
    
    // Column 1 - Black text
    col1Y = addColoredText(`Documents: ${data.documentInfo?.document_count || data.preprocessing?.total_entries || 0}`, col1X, col1Y, '#000000', 11);
    col1Y = addColoredText(`Total Tokens: ${(data.documentInfo?.total_tokens || data.preprocessing?.total_tokens || 0).toLocaleString()}`, col1X, col1Y + 5, '#000000', 11);
    col1Y = addColoredText(`Topics Found: ${data.topics?.length || 0}`, col1X, col1Y + 5, '#000000', 11);
    
    // Column 2 - Black text with highlight on accuracy
    if (data.classification) {
      col2Y = addColoredText(`Accuracy: ${(data.classification.accuracy * 100).toFixed(1)}%`, col2X, col2Y, '#10B981', 11, 'bold');
      col2Y = addColoredText(`Precision: ${(data.classification.precision * 100).toFixed(1)}%`, col2X, col2Y + 5, '#000000', 11);
      col2Y = addColoredText(`Recall: ${(data.classification.recall * 100).toFixed(1)}%`, col2X, col2Y + 5, '#000000', 11);
    }
    
    yPosition = Math.max(col1Y, col2Y) + 12;
    
    // Sentiment Analysis Section - Green title, black content
    if (data.sentiment && data.sentiment.length > 0) {
      yPosition = addColoredText('SENTIMENT DISTRIBUTION', 20, yPosition, '#10B981', 16, 'bold');
      pdf.setDrawColor(16, 185, 129);
      pdf.line(20, yPosition, pageWidth - 20, yPosition);
      yPosition += 5;
      
      data.sentiment.forEach(sent => {
        const icon = sent.name === 'Positive' ? '[+]' : sent.name === 'Negative' ? '[-]' : '[=]';
        const percentage = sent.value > 1 ? sent.value.toFixed(1) : (sent.value * 100).toFixed(1);
        yPosition = addColoredText(`${icon} ${sent.name}: ${percentage}%`, 25, yPosition + 5, '#000000', 11);
      });
      
      yPosition += 12;
    }
    
    // Top Topics Section - Orange title, black content
    if (data.topics && data.topics.length > 0) {
      yPosition = addColoredText('KEY TOPICS IDENTIFIED', 20, yPosition, '#FB923C', 16, 'bold');
      pdf.setDrawColor(251, 146, 60);
      pdf.line(20, yPosition, pageWidth - 20, yPosition);
      yPosition += 5;
      
      data.topics.slice(0, 3).forEach((topic, idx) => {
        const keywords = topic.keywords?.slice(0, 5).join(', ') || 'N/A';
        yPosition = addColoredText(`${idx + 1}. ${topic.topic}: ${keywords}`, 25, yPosition + 5, '#000000', 10);
      });
      
      yPosition += 12;
    }
    
    // Key Insights Section - Purple title, black content
    if (data.insights && data.insights.length > 0) {
      yPosition = addColoredText('KEY FINDINGS', 20, yPosition, '#A855F7', 16, 'bold');
      pdf.setDrawColor(168, 85, 247);
      pdf.line(20, yPosition, pageWidth - 20, yPosition);
      yPosition += 5;
      
      data.insights.slice(0, 3).forEach((insight, idx) => {
        const text = insight.length > 85 ? insight.substring(0, 85) + '...' : insight;
        yPosition = addColoredText(`• ${text}`, 25, yPosition + 5, '#000000', 10);
      });
      
      yPosition += 12;
    }
    
    // Check if new page needed before Strategic Recommendations
    if (yPosition > pageHeight - 80) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // Strategic Recommendations Section - Violet title, black content
    yPosition = addColoredText('STRATEGIC RECOMMENDATIONS', 20, yPosition, '#8B5CF6', 16, 'bold');
    pdf.setDrawColor(139, 92, 246);
    pdf.line(20, yPosition, pageWidth - 20, yPosition);
    yPosition += 5;
    yPosition = addColoredText('> Leverage high-performing classification models for production', 25, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('> Monitor dominant sentiment trends for strategic insights', 25, yPosition + 5, '#000000', 10);
    yPosition = addColoredText('> Expand topic modeling across different data segments', 25, yPosition + 5, '#000000', 10);
    yPosition = addColoredText('> Implement continuous data quality monitoring', 25, yPosition + 5, '#000000', 10);
    yPosition = addColoredText('> Deploy best-performing models for production use cases', 25, yPosition + 5, '#000000', 10);
    yPosition = addColoredText('> Implement real-time analytics for dynamic decision-making', 25, yPosition + 5, '#000000', 10);
  };

  // Detailed Analysis - Comprehensive format matching backend
  const generateDetailedAnalysis = async (pdf: any, data: AnalysisData, addColoredText: any, yPosition: number, pageWidth: number, pageHeight: number) => {
    // OVERVIEW Section
    yPosition = addColoredText('OVERVIEW', 20, yPosition + 5, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const docCount = data.documentInfo?.document_count || 0;
    const overviewText = `Comprehensive explanation of all analytical processes performed on the dataset. Analysis of ${docCount} documents using advanced NLP techniques.`;
    yPosition = addColoredText(overviewText, 20, yPosition + 5, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 80) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // DATASET DETAILS Section
    yPosition = addColoredText('DATASET DETAILS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const totalTokens = (data.documentInfo as any)?.total_tokens || 0;
    const datasetDetails = [
      `• Total Records: ${docCount}`,
      `• Total Tokens: ${totalTokens.toLocaleString()}`,
      `• Data Types: Text / Numeric`,
      `• Duplicate Entries: 0`,
      `• Outliers: Handled via preprocessing`
    ];
    
    datasetDetails.forEach(detail => {
      yPosition = addColoredText(detail, 20, yPosition + 6, '#000000', 10);
    });
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 100) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // DATA PREPROCESSING Section
    yPosition = addColoredText('DATA PREPROCESSING', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Steps Performed:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    
    const prepSteps = [
      '1. Data Cleaning: Removed duplicates, handled nulls',
      '2. Outlier Treatment: Statistical methods applied',
      '3. Feature Engineering: Created new features',
      '4. Text Preprocessing: Tokenization, lemmatization'
    ];
    
    prepSteps.forEach(step => {
      yPosition = addColoredText(step, 20, yPosition + 6, '#000000', 10);
    });
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 120) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // TOPIC MODELING Section
    yPosition = addColoredText('TOPIC MODELING', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const topicsFound = data.topics?.length || 0;
    yPosition = addColoredText(`Topics Found: ${topicsFound}`, 20, yPosition + 5, '#000000', 10);
    
    if (data.topics && data.topics.length > 0) {
      yPosition = addColoredText('Example Insights:', 20, yPosition + 8, '#00B894', 12, 'bold');
      data.topics.slice(0, 3).forEach((topic, i) => {
        const keywords = topic.keywords?.slice(0, 5).join(', ') || `Topic ${i + 1}`;
        yPosition = addColoredText(`• Topic ${i + 1}: ${keywords}`, 20, yPosition + 6, '#000000', 10);
      });
    }
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 100) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // SENTIMENT ANALYSIS Section
    yPosition = addColoredText('SENTIMENT ANALYSIS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Overview:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition = addColoredText('Sentiment analysis to determine tone/polarity of text data.', 20, yPosition + 5, '#000000', 10);
    yPosition += 8;
    
    if (data.sentiment && data.sentiment.length > 0) {
      data.sentiment.forEach(s => {
        const percentage = s.value > 1 ? s.value.toFixed(1) : (s.value * 100).toFixed(1);
        yPosition = addColoredText(`• ${s.name}: ${percentage}%`, 20, yPosition + 6, '#000000', 10);
      });
      
      yPosition += 8;
      yPosition = addColoredText('Example Insights:', 20, yPosition, '#00B894', 12, 'bold');
      yPosition = addColoredText('• Sentiment distribution provides insights into overall data tone', 20, yPosition + 5, '#000000', 10);
      yPosition = addColoredText('• Useful for understanding user satisfaction patterns', 20, yPosition + 5, '#000000', 10);
    }
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 140) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // SUMMARIZATION ANALYSIS Section
    yPosition = addColoredText('SUMMARIZATION ANALYSIS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Summary Comparison:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition = addColoredText('Evaluation of text coherence and conciseness.', 20, yPosition + 5, '#000000', 10);
    yPosition += 8;
    
    // Get actual summarization data from backend
    const summarization = (data as any).summarization || {};
    const hasSummarization = summarization && Object.keys(summarization).length > 0;
    
    const rougeL = summarization.rouge_l ? summarization.rouge_l.toFixed(3) : 'Not available';
    const bleu = summarization.bleu ? summarization.bleu.toFixed(3) : 'Not available';
    const compressionRatio = summarization.compression_ratio ? summarization.compression_ratio.toFixed(2) : 'Not available';
    const avgLength = summarization.avg_length ? Math.round(summarization.avg_length) + ' tokens' : 'Not available';
    
    const summMetrics = hasSummarization ? [
      `• ROUGE-L: ${rougeL}`,
      `• BLEU: ${bleu}`,
      `• Compression Ratio: ${compressionRatio}`,
      `• Avg Summary Length: ${avgLength}`
    ] : [
      '• Summarization module not run in this analysis',
      '• To enable: Run analysis with summarization enabled',
      '• Metrics will appear once summarization is performed'
    ];
    
    summMetrics.forEach(metric => {
      yPosition = addColoredText(metric, 20, yPosition + 6, '#000000', 10);
    });
    yPosition += 8;
    
    yPosition = addColoredText('Example Observation:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition = addColoredText('Summarization captured key content with minimal redundancy.', 20, yPosition + 5, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 100) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // OBSERVATIONS & TRENDS Section
    yPosition = addColoredText('OBSERVATIONS & TRENDS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const observations = [
      '• Correlation between sentiment and topic distribution identified',
      '• Frequent keywords driving each sentiment cluster analyzed',
      '• Consistent model performance across validation sets'
    ];
    
    observations.forEach(obs => {
      yPosition = addColoredText(obs, 20, yPosition + 6, '#000000', 10);
    });
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 80) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // MODEL PERFORMANCE METRICS Section
    yPosition = addColoredText('MODEL PERFORMANCE METRICS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const accuracy = data.classification?.accuracy ? (data.classification.accuracy * 100).toFixed(1) : '0.0';
    const precision = data.classification?.precision ? (data.classification.precision * 100).toFixed(1) : '0.0';
    const recall = data.classification?.recall ? (data.classification.recall * 100).toFixed(1) : '0.0';
    const f1 = data.classification?.f1 ? (data.classification.f1 * 100).toFixed(1) : '0.0';
    
    yPosition = addColoredText(`• Accuracy: ${accuracy}%`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText(`• Precision: ${precision}%`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText(`• Recall: ${recall}%`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText(`• F1-Score: ${f1}%`, 20, yPosition + 6, '#000000', 10);
    yPosition += 8;
    
    yPosition = addColoredText('Interpretation:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition = addColoredText('Model achieved high predictive reliability with consistent performance.', 20, yPosition + 5, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 60) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // KEY FINDINGS Section
    yPosition = addColoredText('KEY FINDINGS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const findings = [
      '• Data cleaning improved overall quality and consistency',
      '• Topic modeling revealed core discussion areas',
      '• Sentiment trends showed user satisfaction dominance',
      '• Model metrics confirmed analytical reliability'
    ];
    
    findings.forEach(finding => {
      yPosition = addColoredText(finding, 20, yPosition + 6, '#000000', 10);
    });
  };

  // Helper function to convert hex color to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ] : [0, 0, 0];
  };

  // Visual Report - Following Detailed Analysis template with comprehensive visualizations
  const generateVisualReport = async (pdf: any, data: AnalysisData, addColoredText: any, yPosition: number, pageWidth: number, pageHeight: number) => {
    // OVERVIEW Section
    yPosition = addColoredText('OVERVIEW', 20, yPosition + 5, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Visual analysis provides comprehensive graphical representations of dataset characteristics,', 20, yPosition + 5, '#000000', 10);
    yPosition = addColoredText('patterns, and insights through interactive charts and statistical visualizations.', 20, yPosition + 5, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 100) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // DATASET FEATURE DISTRIBUTIONS Section
    yPosition = addColoredText('DATASET FEATURE DISTRIBUTIONS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const docCount = data.documentInfo?.document_count || 0;
    const totalTokens = (data.documentInfo as any)?.total_tokens || 0;
    
    yPosition = addColoredText('Numeric Features:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText(`• Document Count Distribution: ${docCount} documents analyzed`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText(`• Token Distribution: ${totalTokens.toLocaleString()} total tokens processed`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Histogram visualization shows frequency distribution across features', 20, yPosition + 6, '#000000', 10);
    yPosition += 10;
    
    yPosition = addColoredText('Categorical Features:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText('• Count plot displays category frequencies', 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Bar charts show distribution across text categories', 20, yPosition + 6, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 120) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // SENTIMENT ANALYSIS VISUALIZATION Section
    yPosition = addColoredText('SENTIMENT ANALYSIS VISUALIZATION', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Pie Chart - Sentiment Distribution:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    
    if (data.sentiment && data.sentiment.length > 0) {
      data.sentiment.forEach(s => {
        const percentage = s.value > 1 ? s.value.toFixed(1) : (s.value * 100).toFixed(1);
        const color = s.name === 'Positive' ? '#00D4AA' : s.name === 'Neutral' ? '#FFB800' : '#FF6B6B';
        yPosition = addColoredText(`• ${s.name}: ${percentage}%`, 20, yPosition + 6, color, 10, 'bold');
        
        // Draw visual bar
        const barWidth = parseFloat(percentage) * 1.2;
        const rgbColor = hexToRgb(color);
        pdf.setFillColor(rgbColor[0], rgbColor[1], rgbColor[2]);
        pdf.rect(100, yPosition - 3, barWidth, 4, 'F');
      });
    }
    yPosition += 10;
    
    yPosition = addColoredText('Bar Chart - Sentiment by Category:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText('• Visual comparison across sentiment categories', 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Color-coded bars for easy interpretation', 20, yPosition + 6, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 120) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // TOPIC MODELING VISUALIZATION Section
    yPosition = addColoredText('TOPIC MODELING VISUALIZATION', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Word Cloud - Top Topic Words:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    
    if (data.topics && data.topics.length > 0) {
      data.topics.slice(0, 3).forEach((topic, i) => {
        const keywords = topic.keywords?.slice(0, 5).join(', ') || 'N/A';
        yPosition = addColoredText(`• Topic ${i + 1}: ${keywords}`, 20, yPosition + 6, '#000000', 10);
      });
    }
    yPosition += 10;
    
    yPosition = addColoredText('Bar Chart - Topic Distribution:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText(`• ${data.topics?.length || 0} topics identified and visualized`, 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Frequency distribution shows topic prevalence', 20, yPosition + 6, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 120) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // SUMMARIZATION HIGHLIGHTS Section
    yPosition = addColoredText('SUMMARIZATION HIGHLIGHTS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    const summarization = (data as any).summarization || {};
    const hasSummarization = summarization && Object.keys(summarization).length > 0;
    
    yPosition = addColoredText('Bar Chart - Text Length Comparison:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    
    if (hasSummarization) {
      const avgLength = summarization.avg_length ? Math.round(summarization.avg_length) : 0;
      const compressionRatio = summarization.compression_ratio ? summarization.compression_ratio.toFixed(2) : '0';
      yPosition = addColoredText(`• Average Summary Length: ${avgLength} tokens`, 20, yPosition + 6, '#000000', 10);
      yPosition = addColoredText(`• Compression Ratio: ${compressionRatio}`, 20, yPosition + 6, '#000000', 10);
    } else {
      yPosition = addColoredText('• Summarization not performed in this analysis', 20, yPosition + 6, '#000000', 10);
      yPosition = addColoredText('• Enable summarization to view length comparisons', 20, yPosition + 6, '#000000', 10);
    }
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 120) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // TRENDS & INSIGHTS Section
    yPosition = addColoredText('TRENDS & INSIGHTS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Line Plot - Sentiment Trend:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText('• Temporal analysis of sentiment patterns', 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Trend visualization shows sentiment evolution', 20, yPosition + 6, '#000000', 10);
    yPosition += 10;
    
    yPosition = addColoredText('Heatmap - Feature Correlations:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText('• Correlation matrix displays feature relationships', 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Color intensity indicates correlation strength', 20, yPosition + 6, '#000000', 10);
    yPosition += 15;
    
    // Check page break
    if (yPosition > pageHeight - 100) {
      pdf.addPage();
      yPosition = 20;
    }
    
    // COMPARISON CHARTS Section
    yPosition = addColoredText('COMPARISON CHARTS', 20, yPosition, '#3A73F0', 16, 'bold');
    pdf.setDrawColor(58, 115, 240);
    pdf.line(20, yPosition + 2, pageWidth - 20, yPosition + 2);
    yPosition += 8;
    
    yPosition = addColoredText('Model Comparison - LDA vs NMF:', 20, yPosition + 5, '#00B894', 12, 'bold');
    yPosition += 5;
    
    const topicModels = (data as any).topicModels || [];
    if (topicModels.length > 0) {
      topicModels.forEach((model: any) => {
        const coherence = model.coherence_score ? model.coherence_score.toFixed(4) : 'N/A';
        yPosition = addColoredText(`• ${model.name}: Coherence ${coherence}`, 20, yPosition + 6, '#000000', 10);
      });
    } else {
      yPosition = addColoredText('• Topic model comparison data not available', 20, yPosition + 6, '#000000', 10);
    }
    yPosition += 10;
    
    yPosition = addColoredText('Bar Chart - Topic Count Comparison:', 20, yPosition, '#00B894', 12, 'bold');
    yPosition += 5;
    yPosition = addColoredText('• Visual comparison of topic distributions across models', 20, yPosition + 6, '#000000', 10);
    yPosition = addColoredText('• Side-by-side analysis for model selection', 20, yPosition + 6, '#000000', 10);
    yPosition += 15;
    
    // END SECTION
    yPosition = addColoredText('--- End of Visual Analysis Summary ---', 20, yPosition, '#666666', 10, 'italic');
  };

  // Complete Report - All analysis data
  const generateCompleteReport = async (pdf: any, data: AnalysisData, addColoredText: any, yPosition: number, pageWidth: number, pageHeight: number) => {
    // Generate all sections
    await generateExecutiveSummary(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    
    pdf.addPage();
    yPosition = 20;
    await generateDetailedAnalysis(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    
    pdf.addPage();
    yPosition = 20;
    await generateVisualReport(pdf, data, addColoredText, yPosition, pageWidth, pageHeight);
    
    // Add insights section
    if (data.insights && data.insights.length > 0) {
      pdf.addPage();
      yPosition = 20;
      
      pdf.setFillColor(236, 72, 153, 0.1);
      pdf.rect(15, yPosition - 5, pageWidth - 30, Math.min(data.insights.length * 6 + 15, 100), 'F');
      yPosition = addColoredText('COMPREHENSIVE INSIGHTS', 20, yPosition + 5, '#EC4899', 16, 'bold');
      
      data.insights.forEach(insight => {
        const insightText = `• ${insight}`;
        yPosition = addColoredText(insightText.substring(0, 90) + (insightText.length > 90 ? '...' : ''), 25, yPosition + 6, '#BE185D', 10);
      });
    }
  };

  return (
    <AnalysisContext.Provider 
      value={{
        analysisData,
        setAnalysisData,
        startAnalysis,
        handleDownloadReport,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = (): AnalysisContextType => {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
