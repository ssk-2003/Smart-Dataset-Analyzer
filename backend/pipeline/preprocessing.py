"""
Preprocessing Pipeline for Text Analysis
"""

import re
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from loguru import logger


class PreprocessingPipeline:
    """
    Text preprocessing pipeline with NLTK and spaCy
    """
    
    def __init__(self):
        """Initialize preprocessing components"""
        try:
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except:
                logger.warning("spaCy model not found, using basic preprocessing")
                self.nlp = None
                
        except Exception as e:
            logger.error(f"Error initializing preprocessing: {str(e)}")
            self.stop_words = set()
            self.nlp = None
    
    def process(self, documents: List[str]) -> Dict[str, Any]:
        """
        Process documents through preprocessing pipeline
        
        Args:
            documents: List of text documents
            
        Returns:
            Dictionary containing preprocessing results
        """
        logger.info(f"Preprocessing {len(documents)} documents...")
        
        # Calculate initial statistics
        initial_stats = self._calculate_initial_stats(documents)
        
        # Process documents
        processed_docs = []
        all_tokens = []
        
        for doc in documents:
            # Clean and tokenize
            tokens = self._tokenize(doc)
            
            # Remove stopwords
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
            
            # Lemmatize
            if self.nlp:
                doc_spacy = self.nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc_spacy]
            else:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
            processed_docs.append(' '.join(tokens))
            all_tokens.extend(tokens)
        
        # Calculate final statistics
        vocab = set(all_tokens)
        token_freq = Counter(all_tokens)
        
        # Identify outliers (very long or very short documents)
        doc_lengths = [len(doc.split()) for doc in processed_docs]
        mean_length = np.mean(doc_lengths)
        std_length = np.std(doc_lengths)
        outliers = sum(1 for l in doc_lengths if abs(l - mean_length) > 2 * std_length)
        
        # Calculate quality metrics
        text_completeness = 1.0 - (initial_stats["missing_percentage"] / 100)
        duplicate_ratio = 1.0 - (initial_stats["duplicate_count"] / len(documents)) if documents else 1.0
        
        results = {
            "total_tokens": len(all_tokens),
            "vocabulary_size": len(vocab),
            "avg_doc_length": mean_length,
            "missing_values": initial_stats["missing_percentage"],
            "duplicates": initial_stats["duplicate_count"],
            "stopwords_removed": initial_stats["original_tokens"] - len(all_tokens),
            "outliers": outliers,
            "processed_documents": processed_docs,
            "token_frequency": dict(token_freq.most_common(100)),
            "statistics": {
                "min_doc_length": min(doc_lengths) if doc_lengths else 0,
                "max_doc_length": max(doc_lengths) if doc_lengths else 0,
                "std_doc_length": std_length
            },
            "quality_metrics": {
                "text_completeness": text_completeness,
                "language_consistency": 1.0,  # Placeholder - can add language detection
                "encoding_quality": 1.0,  # Placeholder - can add encoding validation
                "duplicate_detection": duplicate_ratio
            }
        }
        
        logger.info(f"Preprocessing complete: {len(vocab)} unique tokens from {len(all_tokens)} total")
        
        return results
    
    def _calculate_initial_stats(self, documents: List[str]) -> Dict[str, Any]:
        """Calculate initial document statistics"""
        # Check for empty/missing documents
        empty_count = sum(1 for doc in documents if not doc or len(doc.strip()) == 0)
        missing_percentage = (empty_count / len(documents)) * 100 if documents else 0
        
        # Check for duplicates
        unique_docs = set(documents)
        duplicate_count = len(documents) - len(unique_docs)
        
        # Count original tokens
        original_tokens = sum(len(self._tokenize(doc)) for doc in documents)
        
        return {
            "missing_percentage": missing_percentage,
            "duplicate_count": duplicate_count,
            "original_tokens": original_tokens
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split
            tokens = text.lower().split()
        
        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
    
    def _normalize(self, text: str) -> str:
        """Normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        return text.strip()
