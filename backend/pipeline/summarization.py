"""
Enhanced Summarization Pipeline with Fine-tuning Capabilities
"""

from typing import List, Dict, Any, Optional, Tuple
import nltk
import re
import numpy as np
from collections import Counter
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# Try to import advanced summarization models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using traditional methods")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logger.warning("Textstat not available, using basic readability metrics")

class SummarizationPipeline:
    """
    Enhanced summarization with multiple algorithms and fine-tuning
    """

    def __init__(self, enable_fine_tuning: bool = True):
        """Initialize enhanced summarization components"""
        self.enable_fine_tuning = enable_fine_tuning
        self.transformer_model = None
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass

        # Initialize advanced models if fine-tuning is enabled
        if self.enable_fine_tuning and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()

        logger.info(f"Enhanced SummarizationPipeline initialized (fine-tuning: {enable_fine_tuning})")
    
    def _initialize_transformer_model(self):
        """Initialize lightweight transformer model for abstractive summarization"""
        try:
            # Use a lightweight, fast model for summarization
            model_name = "facebook/bart-large-cnn"  # Good balance of quality and speed
            self.transformer_model = pipeline(
                "summarization",
                model=model_name,
                device=-1,  # Use CPU for compatibility
                framework="pt"
            )
            logger.info("Transformer model initialized for enhanced summarization")
        except Exception as e:
            logger.warning(f"Failed to initialize transformer model: {e}")
            self.transformer_model = None

    def process(self, documents: List[str], summary_sentences: int = 3) -> Dict[str, Any]:
        """
        Enhanced summarization with multiple algorithms and quality optimization
        """
        if not documents:
            return {"summary": "", "key_sentences": [], "document_summaries": []}

        logger.info(f"Starting enhanced summarization for {len(documents)} documents...")

        # Enhanced preprocessing
        processed_docs = self._preprocess_documents(documents)
        
        # Smart sampling for performance
        max_docs_for_summary = 30 if self.enable_fine_tuning else 20
        if len(processed_docs) > max_docs_for_summary:
            import random
            random.seed(42)
            selected_docs = random.sample(processed_docs, max_docs_for_summary)
            logger.info(f"🚀 Sampled {max_docs_for_summary} documents from {len(processed_docs)} for summarization")
        else:
            selected_docs = processed_docs

        # Enhanced text combination with smart truncation
        combined_text = self._combine_texts_intelligently(selected_docs)
        
        # Generate multiple summaries and select the best
        summaries = {}
        
        # Traditional extractive summary (Sumy LSA)
        sumy_summary = self._sumy_summarize(combined_text, summary_sentences)
        summaries["extractive"] = {
            "text": sumy_summary,
            "method": "Sumy LSA",
            "type": "extractive"
        }
        
        # Enhanced abstractive summary (if available)
        if self.enable_fine_tuning and self.transformer_model:
            abstractive_summary = self._transformer_summarize(combined_text, summary_sentences)
            summaries["abstractive"] = {
                "text": abstractive_summary,
                "method": "BART CNN",
                "type": "abstractive"
            }
        
        # Select best summary based on quality metrics
        best_summary = self._select_best_summary(summaries, combined_text)
        
        # Enhanced key insights extraction
        key_insights = self._extract_enhanced_insights(combined_text, 5)
        
        # Enhanced document summaries
        doc_summaries = self._create_enhanced_document_summaries(documents[:5])
        
        # Calculate enhanced statistics
        stats = self._calculate_enhanced_statistics(combined_text, best_summary["text"], summaries)
        
        return {
            "summary": best_summary["text"],
            "extractive_summary": summaries["extractive"]["text"],
            "abstractive_summary": summaries.get("abstractive", {}).get("text", ""),
            "key_sentences": key_insights,
            "document_summaries": doc_summaries,
            "method_used": best_summary["method"],
            "summary_type": best_summary["type"],
            "algorithms_used": list(summaries.keys()),
            "enhancement_enabled": self.enable_fine_tuning,
            **stats
        }
    
    def _preprocess_documents(self, documents: List[str]) -> List[str]:
        """Enhanced preprocessing for better summarization"""
        processed = []
        
        for doc in documents:
            if not doc or len(doc.strip()) < 10:
                continue
            
            # Remove academic formatting artifacts
            doc = re.sub(r'\b\d{4}\s*\d+\s*IJRTI\b.*?(?=\s[A-Z])', '', doc)  # Remove journal headers
            doc = re.sub(r'ISSN:\s*\d+-\d+', '', doc)  # Remove ISSN
            doc = re.sub(r'Volume\s*\d+,?\s*Issue\s*\d+', '', doc)  # Remove volume/issue
            doc = re.sub(r'International Journal.*?Innovation\s*\d*', '', doc)  # Remove journal name
            
            # Clean general formatting
            doc = re.sub(r'\s+', ' ', doc)  # Normalize whitespace
            doc = re.sub(r'([a-z])([A-Z])', r'\1 \2', doc)  # Add space between camelCase
            doc = re.sub(r'(\w)([A-Z][a-z])', r'\1 \2', doc)  # Fix concatenated words
            
            # Remove very short sentences
            sentences = re.split(r'[.!?]+', doc)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            doc = '. '.join(meaningful_sentences)
            
            if len(doc.strip()) >= 20:  # Keep only meaningful content
                processed.append(doc.strip())
        
        logger.info(f"Enhanced preprocessing: {len(documents)} → {len(processed)} documents")
        return processed
    
    def _combine_texts_intelligently(self, documents: List[str]) -> str:
        """Intelligently combine texts with smart truncation"""
        # Sort documents by length and importance
        scored_docs = []
        for doc in documents:
            # Score based on length and content quality
            word_count = len(doc.split())
            sentence_count = len(re.split(r'[.!?]+', doc))
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Prefer documents with good sentence structure
            quality_score = word_count * 0.5 + avg_sentence_length * 0.3 + sentence_count * 0.2
            scored_docs.append((quality_score, doc))
        
        # Sort by quality and take the best documents
        scored_docs.sort(reverse=True)
        
        # Combine texts up to limit
        combined = ""
        max_chars = 15000 if self.enable_fine_tuning else 10000
        
        for score, doc in scored_docs:
            if len(combined) + len(doc) <= max_chars:
                combined += doc + " "
            else:
                # Add partial document if it fits
                remaining_space = max_chars - len(combined)
                if remaining_space > 100:
                    combined += doc[:remaining_space] + "..."
                break
        
        return combined.strip()
    
    def _transformer_summarize(self, text: str, n_sentences: int = 3) -> str:
        """Generate abstractive summary using transformer model"""
        try:
            if not self.transformer_model or len(text.split()) < 10:
                return ""
            
            # Calculate target length
            input_length = len(text.split())
            min_length = max(20, min(50, input_length // 4))
            max_length = max(min_length + 20, min(150, input_length // 2))
            
            # Generate summary
            summary_result = self.transformer_model(
                text[:1024],  # Limit input length for performance
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-process summary
            summary = self._post_process_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Transformer summarization failed: {e}")
            return ""
    
    def _select_best_summary(self, summaries: Dict[str, Dict], original_text: str) -> Dict[str, str]:
        """Select the best summary based on quality metrics"""
        if not summaries:
            return {"text": "", "method": "None", "type": "none"}
        
        best_summary = None
        best_score = 0
        
        for summary_type, summary_data in summaries.items():
            summary_text = summary_data["text"]
            if not summary_text:
                continue
            
            # Calculate quality score
            score = self._calculate_summary_quality(summary_text, original_text)
            
            if score > best_score:
                best_score = score
                best_summary = summary_data
        
        # Fallback to first available summary
        if not best_summary:
            best_summary = list(summaries.values())[0]
        
        return best_summary
    
    def _calculate_summary_quality(self, summary: str, original: str) -> float:
        """Calculate summary quality score"""
        try:
            # Basic metrics
            summary_words = len(summary.split())
            original_words = len(original.split())
            
            if summary_words == 0 or original_words == 0:
                return 0
            
            # Compression ratio (prefer moderate compression)
            compression = 1 - (summary_words / original_words)
            compression_score = 1 - abs(compression - 0.7)  # Target 70% compression
            
            # Length appropriateness
            length_score = 1 if 20 <= summary_words <= 150 else 0.5
            
            # Readability (prefer complete sentences)
            sentence_count = len(re.split(r'[.!?]+', summary.strip()))
            readability_score = min(1, sentence_count / 3)  # Prefer 3+ sentences
            
            # Content coverage (simple word overlap)
            summary_words_set = set(summary.lower().split())
            original_words_set = set(original.lower().split())
            coverage = len(summary_words_set & original_words_set) / len(summary_words_set)
            
            # Combined score
            total_score = (compression_score * 0.3 + length_score * 0.2 + 
                          readability_score * 0.2 + coverage * 0.3)
            
            return total_score
            
        except Exception:
            return 0.5  # Default score
    
    def _post_process_summary(self, summary: str) -> str:
        """Post-process summary for better quality"""
        if not summary:
            return ""
        
        # Remove incomplete sentences at the end
        sentences = re.split(r'[.!?]+', summary)
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 3:  # Keep sentences with at least 3 words
                complete_sentences.append(sentence)
        
        # Rejoin sentences
        result = '. '.join(complete_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _extract_enhanced_insights(self, text: str, n_insights: int = 5) -> List[str]:
        """Extract enhanced key insights with better scoring"""
        try:
            from nltk.tokenize import sent_tokenize
            from nltk.corpus import stopwords
            
            sentences = sent_tokenize(text)
            if len(sentences) <= n_insights:
                return sentences
            
            # Enhanced scoring algorithm
            scored_sentences = []
            stop_words = set(stopwords.words('english'))
            
            for i, sentence in enumerate(sentences):
                sentence = re.sub(r'\s+', ' ', sentence.strip())
                
                if len(sentence.split()) < 5:  # Skip very short sentences
                    continue
                
                words = sentence.lower().split()
                content_words = [w for w in words if w not in stop_words and len(w) > 2]
                
                # Multiple scoring factors
                length_score = min(len(content_words), 20)  # Prefer moderate length
                position_score = max(0, 15 - i)  # Earlier sentences more important
                keyword_score = sum(1 for w in content_words if w in ['artificial', 'intelligence', 'machine', 'learning', 'data', 'analysis', 'research', 'method', 'application'])
                capital_score = sum(1 for c in sentence if c.isupper()) / len(sentence)
                
                total_score = length_score + position_score + keyword_score * 3 + capital_score * 10
                scored_sentences.append((total_score, sentence))
            
            # Sort and return top insights
            scored_sentences.sort(reverse=True)
            return [sentence for score, sentence in scored_sentences[:n_insights]]
            
        except Exception as e:
            logger.warning(f"Enhanced insight extraction failed: {e}")
            return self._extract_key_sentences(text, n_insights)
    
    def _create_enhanced_document_summaries(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Create enhanced document summaries with quality metrics"""
        doc_summaries = []
        
        for i, doc in enumerate(documents):
            if not doc:
                continue
                
            # Clean document
            cleaned_doc = self._preprocess_documents([doc])[0] if self._preprocess_documents([doc]) else doc
            
            # Generate multiple summaries
            sumy_summary = self._sumy_summarize(cleaned_doc, 2)
            
            # Calculate quality metrics
            original_words = len(cleaned_doc.split())
            summary_words = len(sumy_summary.split()) if sumy_summary else 0
            compression = int((1 - (summary_words / original_words)) * 100) if original_words > 0 else 0
            
            doc_summary = {
                "document_id": i,
                "original_length": original_words,
                "summary_length": summary_words,
                "compression_ratio": compression,
                "extractive_summary": sumy_summary,
                "quality_score": self._calculate_summary_quality(sumy_summary, cleaned_doc) if sumy_summary else 0
            }
            
            # Add abstractive summary if available
            if self.enable_fine_tuning and self.transformer_model and original_words > 20:
                abstractive = self._transformer_summarize(cleaned_doc, 2)
                doc_summary["abstractive_summary"] = abstractive
                doc_summary["has_abstractive"] = bool(abstractive)
            
            doc_summaries.append(doc_summary)
        
        return doc_summaries
    
    def _calculate_enhanced_statistics(self, original_text: str, best_summary: str, summaries: Dict) -> Dict[str, Any]:
        """Calculate enhanced statistics with quality metrics"""
        original_word_count = len(original_text.split())
        summary_word_count = len(best_summary.split()) if best_summary else 0
        
        # Enhanced compression ratio calculation
        compression_ratio = int((1 - (summary_word_count / original_word_count)) * 100) if original_word_count > 0 else 0
        
        # Quality metrics
        readability_score = self._calculate_readability(best_summary) if best_summary else 0
        
        # Algorithm performance
        algorithms_used = list(summaries.keys())
        best_algorithm = None
        if summaries:
            best_algorithm = max(summaries.items(), key=lambda x: self._calculate_summary_quality(x[1]["text"], original_text))[0]
        
        return {
            "original_word_count": original_word_count,
            "summary_word_count": summary_word_count,
            "compression_ratio": compression_ratio,
            "readability_score": readability_score,
            "algorithms_tested": len(algorithms_used),
            "best_algorithm": best_algorithm,
            "quality_optimized": self.enable_fine_tuning
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            if TEXTSTAT_AVAILABLE:
                return flesch_reading_ease(text) / 100  # Normalize to 0-1
            else:
                # Simple readability approximation
                words = len(text.split())
                sentences = len(re.split(r'[.!?]+', text.strip()))
                avg_sentence_length = words / max(sentences, 1)
                
                # Prefer moderate sentence length (10-20 words)
                if 10 <= avg_sentence_length <= 20:
                    return 0.8
                elif 5 <= avg_sentence_length <= 25:
                    return 0.6
                else:
                    return 0.4
        except:
            return 0.5

    def _gensim_summarize(self, text: str, n_sentences: int = 3) -> str:
        """
        Summarize text using Gensim TextRank
        """
        try:
            from gensim.summarization import summarize

            # Gensim summarize expects minimum text length
            if len(text.split()) < 10:
                return text if text else ""

            # Generate summary
            summary = summarize(text, word_count=50 * n_sentences)

            return summary.strip() if summary else ""

        except Exception as e:
            logger.warning(f"Gensim summarization failed: {e}")
            return ""

    def _sumy_summarize(self, text: str, n_sentences: int = 3) -> str:
        """
        Summarize text using Sumy LSA
        """
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer

            # Clean and validate text
            text = text.strip()
            if not text:
                return ""
                
            # Sumy expects minimum text length
            word_count = len(text.split())
            if word_count < 10:
                return text

            # Ensure we don't ask for more sentences than available
            sentences = text.split('.')
            actual_sentences = len([s for s in sentences if s.strip()])
            n_sentences = min(n_sentences, max(1, actual_sentences - 1))

            # Parse text
            parser = PlaintextParser.from_string(text, Tokenizer("english"))

            # Create summarizer
            summarizer = LsaSummarizer()

            # Generate summary
            summary_sentences = summarizer(parser.document, n_sentences)

            # Convert to text
            summary = " ".join([str(sentence) for sentence in summary_sentences])

            # Fallback if summary is empty
            if not summary.strip():
                # Return first few sentences as fallback
                fallback_sentences = text.split('.')[:n_sentences]
                summary = '. '.join([s.strip() for s in fallback_sentences if s.strip()]) + '.'

            return summary.strip() if summary else text[:200] + "..."

        except Exception as e:
            logger.warning(f"Sumy summarization failed: {e}")
            # Return a truncated version as fallback
            return text[:200] + "..." if len(text) > 200 else text

    def _extract_key_sentences(self, text: str, n_sentences: int = 5) -> List[str]:
        """
        Extract key sentences based on simple heuristics
        """
        try:
            from nltk.tokenize import sent_tokenize
            from nltk.corpus import stopwords
            import re

            sentences = sent_tokenize(text)

            if len(sentences) <= n_sentences:
                return sentences

            # Simple scoring based on sentence length and position
            scored_sentences = []

            for i, sentence in enumerate(sentences):
                # Remove extra whitespace
                sentence = re.sub(r'\s+', ' ', sentence.strip())

                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue

                # Score based on length (longer sentences often more important)
                length_score = len(sentence.split())

                # Score based on position (earlier sentences often more important)
                position_score = max(0, 10 - i)

                # Score based on capital letters (indicates proper nouns/importance)
                capital_score = sum(1 for c in sentence if c.isupper())

                total_score = length_score + position_score + capital_score

                scored_sentences.append((total_score, sentence))

            # Sort by score and return top sentences
            scored_sentences.sort(reverse=True)
            return [sentence for score, sentence in scored_sentences[:n_sentences]]

        except Exception as e:
            logger.warning(f"Key sentence extraction failed: {e}")
            return []
