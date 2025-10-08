"""
Enhanced Sentiment Analysis Pipeline with Fine-tuning Capabilities
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import nltk
import re
from collections import Counter
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# Try to import advanced sentiment models
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available, using VADER only")

try:
    import transformers
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using traditional methods")

class SentimentAnalysisPipeline:
    """
    Enhanced sentiment analysis with multiple models and fine-tuning
    """

    def __init__(self, enable_fine_tuning: bool = False):
        """Initialize sentiment analyzers with fine-tuning capabilities (optimized for speed)"""
        self.enable_fine_tuning = enable_fine_tuning
        self.vader = None
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.transformers_available = TRANSFORMERS_AVAILABLE
        self.transformer_model = None
        
        # Emotion lexicon for classification
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'excellent', 'fantastic', 'delighted', 'pleased'],
            'sad': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'sorrowful', 'gloomy', 'melancholy'],
            'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'outraged', 'enraged'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'frightened'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned']
        }
        
        # Initialize VADER
        try:
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")

        # Initialize advanced models if fine-tuning is enabled
        if self.enable_fine_tuning:
            self._initialize_advanced_models()
        
        logger.info(f"Enhanced sentiment pipeline initialized (fine-tuning: {enable_fine_tuning})")
    
    def _initialize_advanced_models(self):
        """Initialize advanced sentiment models for fine-tuning"""
        # Initialize lightweight transformer model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight, fast model for sentiment analysis
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True,
                    device=-1  # Use CPU for compatibility
                )
                logger.info("Transformer model initialized for enhanced sentiment analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize transformer model: {e}")
                self.transformer_model = None

    def process(self, documents: List[str]) -> Dict[str, Any]:
        """
        Enhanced sentiment analysis with optional ensemble methods
        """
        logger.info(f"Analyzing sentiment for {len(documents)} documents with {'enhanced' if self.enable_fine_tuning else 'standard'} methods...")

        # Preprocess documents
        processed_docs = self._preprocess_documents(documents)
        
        # Initialize results with emotion classification
        results = {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "compound_scores": [],
            "document_sentiments": [],
            "average_compound": 0,
            "confidence_scores": [],
            "model_agreement": 0,
            "models_used": [],
            "emotions": {
                "joy": 0,
                "sad": 0,
                "angry": 0,
                "fear": 0,
                "surprise": 0
            }
        }

        if not processed_docs:
            return results

        # Analyze each document
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        compound_scores = []
        confidence_scores = []
        document_sentiments = []

        emotion_counts = {'joy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 'surprise': 0}
        
        for doc in processed_docs:
            if not doc or len(doc.strip()) == 0:
                continue

            # Get sentiment analysis (optimized - use VADER only by default)
            if self.enable_fine_tuning:
                sentiment = self._analyze_document_enhanced(doc)
            else:
                sentiment = self._analyze_document(doc)
            
            # Classify emotion
            emotion = self._classify_emotion(doc)
            sentiment['emotion'] = emotion
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
                
            document_sentiments.append(sentiment)

            # Update counts
            if sentiment["label"] == "positive":
                positive_count += 1
            elif sentiment["label"] == "negative":
                negative_count += 1
            else:
                neutral_count += 1

            compound_scores.append(sentiment["compound"])
            confidence_scores.append(sentiment.get("confidence", 0.5))

        # Calculate percentages
        total = len(processed_docs) if processed_docs else 1
        results["positive"] = (positive_count / total) * 100
        results["neutral"] = (neutral_count / total) * 100
        results["negative"] = (negative_count / total) * 100
        results["compound_scores"] = compound_scores
        results["confidence_scores"] = confidence_scores
        results["average_compound"] = np.mean(compound_scores) if compound_scores else 0
        results["average_confidence"] = np.mean(confidence_scores) if confidence_scores else 0
        results["document_sentiments"] = document_sentiments[:100]  # Limit for response size

        # Enhanced statistics with score distribution
        score_distribution = self._calculate_score_distribution(compound_scores)
        
        results["statistics"] = {
            "std_compound": np.std(compound_scores) if compound_scores else 0,
            "min_compound": min(compound_scores) if compound_scores else 0,
            "max_compound": max(compound_scores) if compound_scores else 0,
            "positive_documents": positive_count,
            "neutral_documents": neutral_count,
            "negative_documents": negative_count,
            "high_confidence_docs": sum(1 for c in confidence_scores if c > 0.7),
            "low_confidence_docs": sum(1 for c in confidence_scores if c < 0.3),
            "total_documents_analyzed": len(processed_docs)
        }
        
        # Add detailed score distribution for frontend visualization
        results["score_distribution"] = score_distribution
        results["distribution_ranges"] = {
            "very_negative": sum(1 for s in compound_scores if s <= -0.6),
            "negative": sum(1 for s in compound_scores if -0.6 < s <= -0.1),
            "neutral": sum(1 for s in compound_scores if -0.1 < s < 0.1),
            "positive": sum(1 for s in compound_scores if 0.1 <= s < 0.6),
            "very_positive": sum(1 for s in compound_scores if s >= 0.6)
        }
        
        # Debug information
        logger.info(f"Sentiment breakdown: Pos={positive_count}, Neu={neutral_count}, Neg={negative_count}, Total={len(processed_docs)}")
        logger.info(f"Compound scores range: {min(compound_scores) if compound_scores else 0:.3f} to {max(compound_scores) if compound_scores else 0:.3f}")
        
        # Log score distribution for frontend debugging
        if score_distribution:
            logger.info(f"Score distribution bins: {len([d for d in score_distribution if d['count'] > 0])} non-empty bins")

        # Add emotion percentages
        total_docs = len(processed_docs) if processed_docs else 1
        for emotion, count in emotion_counts.items():
            results["emotions"][emotion] = (count / total_docs) * 100
        
        # Model information
        models_used = ["VADER"]
        if self.enable_fine_tuning:
            if self.textblob_available:
                models_used.append("TextBlob")
            if self.transformer_model:
                models_used.append("RoBERTa")
        
        results["models_used"] = models_used
        results["enhancement_enabled"] = self.enable_fine_tuning

        logger.info(f"Sentiment analysis complete: Positive={results['positive']:.1f}%, Neutral={results['neutral']:.1f}%, Negative={results['negative']:.1f}% (Confidence: {results['average_confidence']:.2f})")
        logger.info(f"Emotion breakdown: Joy={results['emotions']['joy']:.1f}%, Sad={results['emotions']['sad']:.1f}%, Angry={results['emotions']['angry']:.1f}%")

        return results
    
    def _preprocess_documents(self, documents: List[str]) -> List[str]:
        """Enhanced preprocessing for better sentiment analysis"""
        processed = []
        
        for doc in documents:
            if not doc or len(doc.strip()) < 3:
                continue
                
            # Basic cleaning
            doc = re.sub(r'http\S+', '', doc)  # Remove URLs
            doc = re.sub(r'@\w+', '', doc)     # Remove mentions
            doc = re.sub(r'#\w+', '', doc)     # Remove hashtags
            doc = re.sub(r'\s+', ' ', doc).strip()  # Normalize whitespace
            
            # Keep only meaningful documents
            if len(doc.split()) >= 2:  # At least 2 words
                processed.append(doc)
        
        logger.info(f"Preprocessing: {len(documents)} → {len(processed)} documents")
        return processed
    
    def _calculate_score_distribution(self, compound_scores: List[float]) -> List[Dict[str, Any]]:
        """Calculate sentiment score distribution for visualization"""
        if not compound_scores:
            return []
        
        # Create bins for score distribution
        bins = np.arange(-1.0, 1.1, 0.1)  # -1.0 to 1.0 in 0.1 increments
        hist, bin_edges = np.histogram(compound_scores, bins=bins)
        
        distribution = []
        for i in range(len(hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            count = int(hist[i])
            percentage = (count / len(compound_scores)) * 100 if compound_scores else 0
            
            # Determine range category
            if bin_start <= -0.1:
                category = "negative"
            elif bin_start >= 0.1:
                category = "positive"
            else:
                category = "neutral"
            
            distribution.append({
                "range_start": round(bin_start, 1),
                "range_end": round(bin_end, 1),
                "count": count,
                "percentage": round(percentage, 1),
                "category": category
            })
        
        return distribution

    def _analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single document using VADER
        """
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": "neutral",
            "compound": 0,
            "scores": {}
        }

        # VADER Analysis
        if self.vader:
            try:
                vader_scores = self.vader.polarity_scores(text)
                result["scores"]["vader"] = vader_scores
                result["compound"] = vader_scores["compound"]

                # Determine label based on compound score
                if vader_scores["compound"] >= 0.05:
                    result["label"] = "positive"
                elif vader_scores["compound"] <= -0.05:
                    result["label"] = "negative"
                else:
                    result["label"] = "neutral"

            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")

        return result
    
    def _analyze_document_enhanced(self, text: str) -> Dict[str, Any]:
        """
        Enhanced sentiment analysis using ensemble of multiple models
        """
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": "neutral",
            "compound": 0,
            "confidence": 0.5,
            "scores": {},
            "model_predictions": {}
        }

        predictions = []
        model_scores = {}

        # VADER Analysis
        if self.vader:
            try:
                vader_scores = self.vader.polarity_scores(text)
                result["scores"]["vader"] = vader_scores
                vader_compound = vader_scores["compound"]
                
                if vader_compound >= 0.05:
                    vader_label = "positive"
                elif vader_compound <= -0.05:
                    vader_label = "negative"
                else:
                    vader_label = "neutral"
                
                predictions.append(vader_label)
                model_scores["vader"] = {"label": vader_label, "score": abs(vader_compound)}
                result["model_predictions"]["vader"] = vader_label
                
            except Exception as e:
                logger.debug(f"VADER analysis failed: {e}")

        # TextBlob Analysis
        if self.textblob_available:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    textblob_label = "positive"
                elif polarity < -0.1:
                    textblob_label = "negative"
                else:
                    textblob_label = "neutral"
                
                predictions.append(textblob_label)
                model_scores["textblob"] = {"label": textblob_label, "score": abs(polarity)}
                result["model_predictions"]["textblob"] = textblob_label
                result["scores"]["textblob"] = {"polarity": polarity, "subjectivity": blob.sentiment.subjectivity}
                
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")

        # Transformer Analysis (lightweight)
        if self.transformer_model and len(text.split()) <= 100:  # Limit for performance
            try:
                transformer_result = self.transformer_model(text[:512])  # Limit text length
                
                if transformer_result and len(transformer_result) > 0:
                    scores = transformer_result[0]
                    best_prediction = max(scores, key=lambda x: x['score'])
                    
                    # Map transformer labels to our format
                    label_mapping = {
                        'LABEL_0': 'negative',
                        'LABEL_1': 'neutral', 
                        'LABEL_2': 'positive',
                        'NEGATIVE': 'negative',
                        'NEUTRAL': 'neutral',
                        'POSITIVE': 'positive'
                    }
                    
                    transformer_label = label_mapping.get(best_prediction['label'], 'neutral')
                    predictions.append(transformer_label)
                    model_scores["transformer"] = {"label": transformer_label, "score": best_prediction['score']}
                    result["model_predictions"]["transformer"] = transformer_label
                    result["scores"]["transformer"] = {score['label']: score['score'] for score in scores}
                    
            except Exception as e:
                logger.debug(f"Transformer analysis failed: {e}")

        # Ensemble decision
        if predictions:
            # Majority voting with confidence weighting
            label_counts = Counter(predictions)
            final_label = label_counts.most_common(1)[0][0]
            
            # Calculate confidence based on agreement and individual model confidence
            agreement_ratio = label_counts[final_label] / len(predictions)
            avg_model_confidence = np.mean([scores["score"] for scores in model_scores.values()])
            
            confidence = (agreement_ratio * 0.6) + (avg_model_confidence * 0.4)
            
            result["label"] = final_label
            result["confidence"] = min(0.95, max(0.05, confidence))  # Clamp between 0.05 and 0.95
            
            # Calculate ensemble compound score
            if "vader" in model_scores:
                result["compound"] = result["scores"]["vader"]["compound"]
            else:
                # Fallback compound score calculation
                if final_label == "positive":
                    result["compound"] = confidence * 0.5
                elif final_label == "negative":
                    result["compound"] = -confidence * 0.5
                else:
                    result["compound"] = 0
        
        result["ensemble_agreement"] = len(set(predictions)) == 1 if predictions else False
        result["models_count"] = len(predictions)
        
        return result

    def get_sentiment_summary(self, documents: List[str]) -> str:
        """
        Generate a summary of sentiment analysis
        """
        results = self.process(documents)

        summary = f"""
        Sentiment Analysis Summary:
        - Positive: {results['positive']:.1f}%
        - Neutral: {results['neutral']:.1f}%
        - Negative: {results['negative']:.1f}%
        - Average Compound Score: {results['average_compound']:.3f}
        """

        if results['positive'] > 60:
            summary += "\nOverall sentiment is predominantly positive."
        elif results['negative'] > 40:
            summary += "\nOverall sentiment shows significant negative feedback."
        else:
            summary += "\nOverall sentiment is mixed."

        return summary
    
    def _classify_emotion(self, text: str) -> str:
        """Fast emotion classification using keyword matching"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if not emotion_scores:
            # Default based on sentiment (more sensitive thresholds)
            if self.vader:
                vader_scores = self.vader.polarity_scores(text)
                compound = vader_scores['compound']
                # Lowered thresholds for better detection
                if compound > 0.1:  # Was 0.3
                    return 'joy'
                elif compound < -0.1:  # Was -0.3
                    return 'sad'
                elif abs(compound) < 0.05:
                    return 'neutral'
            return 'neutral'
        
        # Return emotion with highest score
        return max(emotion_scores, key=emotion_scores.get)
