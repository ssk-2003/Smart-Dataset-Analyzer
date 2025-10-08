"""
Utility functions for the Cosmic Analyze Backend
"""

import os
import json
import re
import chardet
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import PyPDF2
import docx


def load_file(file_path: str, text_column: str = "text") -> Tuple[List[str], Dict[str, Any]]:
    """
    Load file and extract text content
    
    Args:
        file_path: Path to the uploaded file
        text_column: Column name for text data (for CSV files)
    
    Returns:
        Tuple of (documents list, metadata dict)
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    documents = []
    metadata = {"file_type": extension}
    
    try:
        if extension == '.csv':
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='warn')
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load with {encoding}: {str(e)}")
            
            if df is None:
                # If all else fails, try with error_bad_lines=False
                logger.warning("All encodings failed, trying with error_bad_lines=False")
                df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', error_bad_lines=False)
            
            # Find text column or a sensible fallback
            if text_column in df.columns:
                documents = df[text_column].dropna().astype(str).tolist()
            else:
                # Prefer common text-like columns often seen in news datasets
                preferred_cols = [
                    c for c in df.columns
                    if c.lower() in {
                        'headline', 'title', 'summary', 'description', 'content', 'news', 'article', 'body'
                    }
                ]

                if preferred_cols:
                    # Use the first preferred column available
                    sel = preferred_cols[0]
                    documents = df[sel].dropna().astype(str).tolist()
                else:
                    # Try columns containing text-like keywords
                    text_like = [
                        col for col in df.columns
                        if ('text' in col.lower()) or ('content' in col.lower()) or ('desc' in col.lower())
                    ]
                    if text_like:
                        documents = df[text_like[0]].dropna().astype(str).tolist()
                    else:
                        # As a robust fallback, combine all string/object columns per row
                        obj_cols = [col for col in df.columns if df[col].dtype == 'object']
                        if obj_cols:
                            combined = (
                                df[obj_cols]
                                .fillna('')
                                .astype(str)
                                .apply(lambda r: '. '.join([v for v in r.tolist() if v and v.strip()]), axis=1)
                            )
                            documents = [d for d in combined.tolist() if d and d.strip()]
                        else:
                            documents = []
            
            # Extract labels if available
            label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]
            if label_cols:
                metadata["labels"] = df[label_cols[0]].tolist()
                
        elif extension == '.txt':
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logger.info(f"Successfully loaded text file with {encoding} encoding")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load with {encoding}: {str(e)}")
            
            if content is None:
                # If all else fails, try with chardet
                try:
                    with open(file_path, 'rb') as f:
                        raw = f.read()
                        result = chardet.detect(raw)
                        detected_encoding = result['encoding'] or 'latin1'
                        content = raw.decode(detected_encoding, errors='replace')
                        logger.info(f"Used chardet to detect encoding: {detected_encoding}")
                except Exception as e:
                    logger.error(f"Failed to load file with any encoding: {str(e)}")
                    raise
            
            # Process the content we already loaded
            if content:
                # Split by paragraphs or sentences
                documents = [p.strip() for p in content.split('\n\n') if p.strip()]
                if len(documents) < 5:
                    # If few paragraphs, split by sentences
                    documents = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
                    
        elif extension == '.pdf':
            # Extract text from PDF
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Split page text into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        documents.extend(paragraphs)
                        
        elif extension in ['.doc', '.docx']:
            # Extract text from Word document
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    documents.append(para.text.strip())
        
        # If still no documents, create a sample
        if not documents:
            documents = [
                "This is a sample document for analysis.",
                "The system will analyze this text for various insights.",
                "Natural language processing techniques will be applied.",
                "Machine learning models will extract patterns and topics.",
                "Results will include sentiment analysis and classification."
            ]
        
        # Clean documents
        documents = [clean_text(doc) for doc in documents if doc and len(doc) > 10]
        
        logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
        return documents, metadata
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        # Return sample data as fallback
        return [
            "Error loading file. Using sample data for demonstration.",
            "This is sample text for analysis pipeline.",
            "The analysis will proceed with this demo content."
        ], metadata


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()


def save_results(results: Dict[str, Any], filename: str) -> str:
    """
    Save analysis results to JSON file
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{filename}_{timestamp}_results.json"
    
    # Convert numpy types to Python types for JSON serialization
    results_json = json.dumps(results, default=str, indent=2)
    
    with open(results_file, 'w') as f:
        f.write(results_json)
    
    logger.info(f"Results saved to {results_file}")
    return str(results_file)


def generate_insights(results: Dict[str, Any], documents: List[str]) -> Dict[str, Any]:
    """
    Generate natural language insights from analysis results
    """
    insights = {
        "text": [],
        "top_terms": []
    }
    
    # Preprocessing insights
    if "preprocessing" in results:
        prep = results["preprocessing"]
        insights["text"].append(
            f"Analyzed {prep['total_tokens']} tokens across {len(documents)} documents with a vocabulary of {prep['vocabulary_size']} unique terms."
        )
        
        if prep["missing_values"] > 10:
            insights["text"].append(
                f"Data quality issue: {prep['missing_values']}% missing values detected. Consider data cleaning."
            )
        
        if prep["duplicates"] > 20:
            insights["text"].append(
                f"Found {prep['duplicates']} duplicate entries. Removing duplicates might improve analysis quality."
            )
    
    # Topic modeling insights
    if "topic_models" in results and results["topic_models"]:
        model = results["topic_models"][0]
        insights["text"].append(
            f"Identified {len(model['topics'])} main topics using {model['name']} model."
        )
        
        # Find dominant topic
        dominant_topic = max(model["topics"], key=lambda x: x["distribution"])
        insights["text"].append(
            f"The dominant topic '{dominant_topic['topic']}' covers {dominant_topic['distribution']*100:.1f}% of the content."
        )
    
    # Sentiment insights
    if "sentiment" in results:
        sent = results["sentiment"]
        if sent["positive"] > 60:
            insights["text"].append(
                f"Overall sentiment is predominantly positive ({sent['positive']:.1f}%), indicating favorable content tone."
            )
        elif sent["negative"] > 40:
            insights["text"].append(
                f"Significant negative sentiment detected ({sent['negative']:.1f}%). Consider reviewing critical feedback."
            )
        else:
            insights["text"].append(
                f"Balanced sentiment distribution with {sent['neutral']:.1f}% neutral content."
            )
    
    # Classification insights
    if "classification" in results:
        clf = results["classification"]
        if clf["accuracy"] > 0.9:
            insights["text"].append(
                f"High classification accuracy ({clf['accuracy']:.2%}) indicates clear patterns in the data."
            )
        
        if clf["recall"] < 0.7:
            insights["text"].append(
                f"Low recall ({clf['recall']:.2%}) suggests the model might be missing important cases."
            )
    
    # Extract top terms from documents
    from collections import Counter
    import re
    
    # Simple term extraction
    all_text = ' '.join(documents[:100])  # Limit for performance
    words = re.findall(r'\b[a-z]+\b', all_text.lower())
    
    # Filter common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
    words = [w for w in words if w not in stopwords and len(w) > 3]
    
    word_freq = Counter(words).most_common(20)
    insights["top_terms"] = [{"term": term, "score": count/len(words)} for term, count in word_freq]
    
    # Add recommendation
    insights["text"].append(
        "Recommendation: Focus on the identified key topics and address any negative sentiment areas for improved outcomes."
    )
    
    return insights


def calculate_text_statistics(documents: List[str]) -> Dict[str, Any]:
    """
    Calculate various text statistics
    """
    import textstat
    
    stats = {
        "total_documents": len(documents),
        "avg_length": np.mean([len(doc.split()) for doc in documents]),
        "min_length": min([len(doc.split()) for doc in documents]) if documents else 0,
        "max_length": max([len(doc.split()) for doc in documents]) if documents else 0,
        "readability": {
            "flesch_reading_ease": np.mean([textstat.flesch_reading_ease(doc) for doc in documents[:100]]),
            "flesch_kincaid_grade": np.mean([textstat.flesch_kincaid_grade(doc) for doc in documents[:100]])
        }
    }
    
    return stats
