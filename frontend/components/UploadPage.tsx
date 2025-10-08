import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Upload, File, CheckCircle, AlertCircle, ArrowLeft, FileText } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { uploadFile as apiUploadFile, checkApiHealth, getAvailableResults } from '@/lib/api';

interface UploadPageProps {
  onBack: () => void;
  onUploadComplete: (file: File | null, filename?: string) => void;
}

interface FileValidation {
  isValid: boolean;
  error?: string;
}

const UploadPage: React.FC<UploadPageProps> = ({ onBack, onUploadComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const { toast } = useToast();

  const ACCEPTED_TYPES = ['.csv', '.doc', '.docx', '.pdf', '.txt'];
  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

  // Check backend connection on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await checkApiHealth();
        if (response.status === 'success') {
          setBackendStatus('connected');
          toast({
            title: "Backend Connected",
            description: "Ready to upload and analyze files",
          });
        } else {
          setBackendStatus('error');
          toast({
            title: "Backend Not Available",
            description: "Please ensure the backend is running on port 8001",
            variant: "destructive"
          });
        }
      } catch (error) {
        setBackendStatus('error');
      }
    };

    checkBackend();
  }, [toast]);

  const validateFile = (file: File): FileValidation => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!ACCEPTED_TYPES.includes(extension)) {
      return {
        isValid: false,
        error: `File type not supported. Please upload: ${ACCEPTED_TYPES.join(', ')}`
      };
    }

    if (file.size > MAX_FILE_SIZE) {
      return {
        isValid: false,
        error: `File size exceeds 50MB limit. Current size: ${(file.size / 1024 / 1024).toFixed(2)}MB`
      };
    }

    return { isValid: true };
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleFile = (file: File) => {
    const validation = validateFile(file);
    
    if (!validation.isValid) {
      toast({
        title: "Invalid File",
        description: validation.error,
        variant: "destructive"
      });
      return;
    }

    setSelectedFile(file);
    setUploadError(null);
    setUploadComplete(false);
    toast({
      title: "File Selected",
      description: `${file.name} is ready for upload`
    });
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const uploadFile = async () => {
    if (!selectedFile || backendStatus !== 'connected') {
      if (backendStatus !== 'connected') {
        toast({
          title: "Backend Not Connected",
          description: "Please start the backend server first",
          variant: "destructive"
        });
      }
      return;
    }
    
    setIsUploading(true);
    setUploadError(null);
    setUploadProgress(0);
    
    try {
      // Upload file with progress tracking
      const response = await apiUploadFile(selectedFile, (progress) => {
        setUploadProgress(progress);
      });
      
      if (response.status === 'error' || !response.data) {
        throw new Error(response.error || 'Upload failed');
      }
      
      setUploadProgress(100);
      setIsUploading(false);
      setUploadComplete(true);
      
      // Store the uploaded filename for analysis
      const uploadedFilename = response.data.filename;
      localStorage.setItem('uploadedFile', uploadedFilename);
      
      toast({
        title: "Upload Successful ✅",
        description: `${selectedFile.name} has been uploaded and is ready for analysis.`,
      });
      
      // Transition to analysis page after a short delay
      setTimeout(() => {
        onUploadComplete(selectedFile, uploadedFilename);
      }, 1500);
    } catch (error) {
      setIsUploading(false);
      setUploadProgress(0);
      
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setUploadError(errorMessage);
      
      toast({
        title: "Upload Failed",
        description: errorMessage,
        variant: "destructive"
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-cosmic relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-secondary rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              scale: [0, 1, 0],
              opacity: [0, 0.8, 0],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              repeatDelay: Math.random() * 4
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div 
          className="flex items-center justify-between mb-12"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Button variant="ghost" onClick={onBack} className="text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              backendStatus === 'connected' ? 'bg-green-500' : 
              backendStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
            } animate-pulse`} />
            <span className="text-sm text-muted-foreground">
              {backendStatus === 'connected' ? 'Backend Connected' : 
               backendStatus === 'error' ? 'Backend Offline' : 'Checking...'}
            </span>
          </div>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-neon bg-clip-text text-transparent mb-4">
              Upload Your Dataset
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Upload your data file and let our AI analyze it for insights, patterns, and predictions.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="bg-card/50 backdrop-blur-sm border-card-border">
              <CardHeader>
                <CardTitle className="text-center text-2xl text-foreground">
                  Select or Drop Your File
                </CardTitle>
              </CardHeader>
              <CardContent className="p-8">
                <AnimatePresence mode="wait">
                  {!uploadComplete ? (
                    <motion.div
                      key="upload"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      {/* Drop Zone */}
                      <div
                        className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 ${
                          dragActive 
                            ? 'border-primary bg-primary/10 shadow-glow-primary/20' 
                            : 'border-card-border hover:border-primary/50 hover:bg-card/30'
                        }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                      >
                        <Upload className={`w-16 h-16 mx-auto mb-6 transition-all duration-300 ${
                          dragActive ? 'text-primary animate-float' : 'text-muted-foreground'
                        }`} />
                        
                        <h3 className="text-xl font-semibold mb-2 text-foreground">
                          {dragActive ? 'Drop your file here' : 'Drag & drop your file here'}
                        </h3>
                        <p className="text-muted-foreground mb-6">
                          Or click to browse and select a file
                        </p>
                        
                        <input
                          type="file"
                          id="file-upload"
                          className="hidden"
                          accept={ACCEPTED_TYPES.join(',')}
                          onChange={handleFileInput}
                        />
                        <Button variant="premium" asChild disabled={backendStatus !== 'connected'}>
                          <label htmlFor="file-upload" className="cursor-pointer">
                            Browse Files
                          </label>
                        </Button>
                      </div>

                      {/* File Info */}
                      {selectedFile && (
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-8 p-6 bg-background-secondary rounded-lg border border-card-border"
                        >
                          <div className="flex items-center gap-4 mb-4">
                            <File className="w-8 h-8 text-primary" />
                            <div className="flex-1">
                              <h4 className="font-semibold text-foreground">{selectedFile.name}</h4>
                              <p className="text-sm text-muted-foreground">
                                {formatFileSize(selectedFile.size)}
                              </p>
                            </div>
                            <CheckCircle className="w-6 h-6 text-green-500" />
                          </div>

                          {uploadError && (
                            <div className="p-4 mb-4 bg-destructive/10 text-destructive rounded-md flex items-center gap-2">
                              <AlertCircle className="w-5 h-5 flex-shrink-0" />
                              <p>{uploadError}</p>
                            </div>
                          )}

                          {isUploading && (
                            <div className="mb-4">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-muted-foreground">Uploading...</span>
                                <span className="text-sm font-semibold text-primary">{uploadProgress}%</span>
                              </div>
                              <Progress value={uploadProgress} className="h-2" />
                            </div>
                          )}

                          <Button
                            variant="hero"
                            onClick={uploadFile}
                            disabled={isUploading || backendStatus !== 'connected'}
                            className="w-full"
                          >
                            {isUploading ? 'Uploading...' : 'Upload & Start Analysis'}
                          </Button>
                        </motion.div>
                      )}

                      {/* File Requirements - Centered */}
                      <div className="mt-8 p-6 bg-card/30 rounded-lg text-center">
                        <h4 className="font-medium mb-3 text-foreground text-lg">Supported File Types:</h4>
                        <p className="text-base text-muted-foreground mb-6 font-medium">
                          CSV, DOC, DOCX, PDF, TXT (Max 50MB)
                        </p>
                        <h4 className="font-medium mb-3 text-foreground text-lg">Tips for Best Results:</h4>
                        <div className="text-sm text-muted-foreground space-y-2 max-w-md mx-auto">
                          <p>• Ensure your text data is clean and well-formatted</p>
                          <p>• For CSV files, include headers for better analysis</p>
                          <p>• Larger datasets provide more accurate insights</p>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="success"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-center py-12"
                    >
                      <CheckCircle className="w-24 h-24 text-green-500 mx-auto mb-6" />
                      <h3 className="text-2xl font-bold text-foreground mb-2">
                        Upload Complete!
                      </h3>
                      <p className="text-muted-foreground mb-6">
                        Your file has been successfully uploaded and is being processed.
                      </p>
                      <div className="animate-pulse text-primary">
                        Redirecting to analysis...
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;