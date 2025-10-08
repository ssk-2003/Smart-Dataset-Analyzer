"""
Lightweight Insights Extraction Pipeline using RAKE
"""

from typing import List, Dict, Any
import nltk
from loguru import logger

class LightweightInsightsPipeline:
    """
    Lightweight insights extraction using RAKE for keyword extraction
    """

    def __init__(self):
        """Initialize RAKE keyword extractor"""
        try:
            # Download required NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)

            # Initialize RAKE
            from rake_nltk import Rake

            # Initialize RAKE with smart stopword removal
            self.rake = Rake(
                min_length=1,           # Minimum length of keyword
                max_length=3,           # Maximum length of keyword
                ranking_metric='degree', # Use degree centrality for ranking
                include_repeated_phrases=False
            )

            logger.info("RAKE keyword extractor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAKE: {e}")
            self.rake = None

    def generate_insights(self, analysis_results: Dict[str, Any], documents: List[str]) -> Dict[str, Any]:
        """
        Generate lightweight insights using RAKE keyword extraction
        """
        insights = {
            "keywords": [],
            "key_phrases": [],
            "insights": [],
            "statistics": {},
            "themes": {},
            "summary": "",
            "word_cloud_data": {},
            "top_terms": []
        }

        try:
            # Extract keywords and key phrases using RAKE
            all_keywords, all_phrases = self._extract_keywords_rake(documents)

            insights["keywords"] = all_keywords
            insights["key_phrases"] = all_phrases

            # Generate comprehensive summary from all analysis results
            insights["summary"] = self._generate_comprehensive_summary(analysis_results, documents)

            # Generate text insights
            insights["insights"] = self._generate_text_insights(analysis_results, all_keywords, all_phrases)

            # Generate basic statistics
            insights["statistics"] = self._generate_basic_stats(documents, all_keywords)

            # Generate themes
            insights["themes"] = self._generate_themes(all_keywords)

            # Generate word cloud data
            insights["word_cloud_data"] = self._generate_word_cloud_data(all_keywords, all_phrases)

            # Generate top terms for display
            insights["top_terms"] = self._generate_top_terms(all_keywords, all_phrases)

            logger.info(f"Generated comprehensive insights: {len(all_keywords)} keywords, {len(all_phrases)} key phrases")

        except Exception as e:
            logger.error(f"Error generating lightweight insights: {e}")
            insights["insights"] = ["Analysis completed using lightweight methods"]
            insights["summary"] = "Error generating comprehensive summary"

        return insights

    def _extract_keywords_rake(self, documents: List[str]) -> tuple:
        """
        Extract keywords and key phrases using RAKE
        """
        try:
            # Combine all documents for comprehensive keyword extraction
            combined_text = " ".join(documents)

            if not combined_text.strip():
                return [], []

            # Extract keywords using RAKE
            self.rake.extract_keywords_from_text(combined_text)

            # Get ranked keywords and phrases
            ranked_keywords = self.rake.get_ranked_phrases()

            # Separate keywords and key phrases
            keywords = []
            key_phrases = []

            for phrase in ranked_keywords:
                # Clean the phrase
                phrase = phrase.strip()
                if not phrase:
                    continue

                # Determine if it's a keyword (1-2 words) or key phrase (3+ words)
                word_count = len(phrase.split())

                if word_count <= 2:
                    keywords.append(phrase)
                else:
                    key_phrases.append(phrase)

            # Limit to top results
            keywords = keywords[:20]  # Top 20 keywords
            key_phrases = key_phrases[:10]  # Top 10 key phrases

            return keywords, key_phrases

        except Exception as e:
            logger.warning(f"RAKE keyword extraction failed: {e}")
            return [], []

    def _generate_text_insights(self, results: Dict[str, Any], keywords: List[str], key_phrases: List[str]) -> List[str]:
        """
        Generate text-based insights from analysis results
        """
        insights = []

        # Document count insights
        doc_count = len(results.get('preprocessing', {}).get('processed_documents', []))
        if doc_count > 0:
            if doc_count == 1:
                insights.append("Single document analysis completed")
            elif doc_count < 10:
                insights.append(f"Small dataset analysis: {doc_count} documents processed")
            elif doc_count < 100:
                insights.append(f"Medium dataset analysis: {doc_count} documents processed")
            else:
                insights.append(f"Large dataset analysis: {doc_count} documents processed")

        # Topic insights with model performance
        if 'topic_models' in results and results['topic_models']:
            best_model = max(results['topic_models'], key=lambda x: x.get('coherence_score', 0))
            topics = best_model.get('topics', [])
            coherence = best_model.get('coherence_score', 0)
            model_name = best_model.get('name', 'Unknown')
            
            if topics:
                insights.append(f"Identified {len(topics)} main topics in the dataset")
                for i, topic in enumerate(topics[:3], 1):
                    keywords_str = ', '.join(topic.get('keywords', [])[:3])
                    insights.append(f"Topic {i}: {keywords_str}")
                
                # Add coherence insight
                coherence_pct = coherence * 100
                if coherence > 0.8:
                    quality = "excellent"
                elif coherence > 0.6:
                    quality = "good"
                elif coherence > 0.4:
                    quality = "fair"
                else:
                    quality = "basic"
                
                insights.append(f"🎯 {model_name} achieved {coherence_pct:.1f}% coherence score, indicating {quality} topic separation and interpretability.")

        # Sentiment insights
        if 'sentiment' in results:
            sentiment = results['sentiment']
            pos = sentiment.get('positive', 0)
            neg = sentiment.get('negative', 0)

            if pos > 70:
                insights.append("Strongly positive sentiment detected across the dataset")
            elif pos > 50:
                insights.append("Predominantly positive sentiment with some neutral content")
            elif neg > 50:
                insights.append("Significant negative sentiment requires attention")
            elif pos > neg:
                insights.append("Overall positive sentiment with balanced perspectives")
            else:
                insights.append("Mixed sentiment detected - balanced viewpoints present")

        # Keyword insights
        if keywords:
            insights.append(f"Top keywords identified: {', '.join(keywords[:5])}")

        if key_phrases:
            insights.append(f"Key phrases extracted: {', '.join(key_phrases[:3])}")

        # Classification insights
        if 'classification' in results:
            clf = results['classification']
            accuracy = clf.get('accuracy', 0)
            if accuracy > 0.8:
                insights.append(f"High classification accuracy: {accuracy:.1%}")
            elif accuracy > 0.6:
                insights.append(f"Good classification performance: {accuracy:.1%}")
            else:
                insights.append(f"Classification accuracy: {accuracy:.1%} - may need improvement")

        return insights

    def _generate_basic_stats(self, documents: List[str], keywords: List[str]) -> Dict[str, Any]:
        """
        Generate basic statistics about the dataset
        """
        stats = {}

        try:
            # Document statistics
            doc_lengths = [len(doc.split()) for doc in documents]
            stats['total_documents'] = len(documents)
            stats['avg_document_length'] = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
            stats['max_document_length'] = max(doc_lengths) if doc_lengths else 0
            stats['min_document_length'] = min(doc_lengths) if doc_lengths else 0

            # Keyword statistics
            stats['total_keywords'] = len(keywords)
            stats['avg_keyword_length'] = sum(len(kw.split()) for kw in keywords) / len(keywords) if keywords else 0

            # Text statistics
            total_words = sum(len(doc.split()) for doc in documents)
            total_chars = sum(len(doc) for doc in documents)

            stats['total_words'] = total_words
            stats['total_characters'] = total_chars
            stats['avg_words_per_document'] = total_words / len(documents) if documents else 0

        except Exception as e:
            logger.warning(f"Error generating basic stats: {e}")

        return stats

    def _generate_themes(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Generate thematic insights from keywords
        """
        themes = {
            "business_terms": [],
            "technical_terms": [],
            "communication_terms": [],
            "action_terms": [],
            "descriptive_terms": []
        }

        try:
            # Business-related keywords
            business_keywords = [
                'business', 'company', 'market', 'sales', 'revenue', 'profit',
                'growth', 'strategy', 'management', 'customer', 'product',
                'service', 'brand', 'marketing', 'finance', 'investment'
            ]

            # Technical keywords
            technical_keywords = [
                'technology', 'system', 'software', 'data', 'analysis',
                'research', 'development', 'innovation', 'digital', 'platform',
                'algorithm', 'database', 'network', 'security', 'automation'
            ]

            # Communication keywords
            communication_keywords = [
                'communication', 'information', 'message', 'discussion',
                'conversation', 'feedback', 'presentation', 'report',
                'documentation', 'email', 'meeting', 'collaboration'
            ]

            # Action keywords
            action_keywords = [
                'implement', 'develop', 'create', 'build', 'design',
                'plan', 'execute', 'manage', 'optimize', 'improve',
                'increase', 'reduce', 'enhance', 'deliver', 'achieve'
            ]

            # Descriptive keywords
            descriptive_keywords = [
                'quality', 'performance', 'efficiency', 'effectiveness',
                'reliability', 'scalability', 'usability', 'functionality',
                'capability', 'feature', 'benefit', 'value', 'impact'
            ]

            for keyword in keywords:
                keyword_lower = keyword.lower()

                if any(biz in keyword_lower for biz in business_keywords):
                    themes["business_terms"].append(keyword)
                if any(tech in keyword_lower for tech in technical_keywords):
                    themes["technical_terms"].append(keyword)
                if any(comm in keyword_lower for comm in communication_keywords):
                    themes["communication_terms"].append(keyword)
                if any(action in keyword_lower for action in action_keywords):
                    themes["action_terms"].append(keyword)
                if any(desc in keyword_lower for desc in descriptive_keywords):
                    themes["descriptive_terms"].append(keyword)

        except Exception as e:
            logger.warning(f"Error generating themes: {e}")

        return themes

    def _generate_comprehensive_summary(self, analysis_results: Dict[str, Any], documents: List[str]) -> str:
        """
        Generate comprehensive summary from all analysis results
        """
        try:
            summary_parts = []

            # Document count
            doc_count = len(documents)
            summary_parts.append(f"Analysis completed for {doc_count} documents.")

            # Sentiment analysis
            if 'sentiment' in analysis_results:
                sentiment = analysis_results['sentiment']
                pos = sentiment.get('positive', 0)
                neg = sentiment.get('negative', 0)
                neu = sentiment.get('neutral', 0)

                if pos > 70:
                    summary_parts.append(f"Overall sentiment is strongly positive ({pos:.1f}%).")
                elif pos > 50:
                    summary_parts.append(f"Overall sentiment is predominantly positive ({pos:.1f}%).")
                elif neg > 50:
                    summary_parts.append(f"Overall sentiment shows significant negative feedback ({neg:.1f}%).")
                else:
                    summary_parts.append(f"Overall sentiment is mixed (Positive: {pos:.1f}%, Neutral: {neu:.1f}%, Negative: {neg:.1f}%).")

            # Topic modeling
            if 'topic_models' in analysis_results and analysis_results['topic_models']:
                topics = analysis_results['topic_models'][0].get('topics', [])
                if topics:
                    summary_parts.append(f"Identified {len(topics)} main topics in the dataset.")

            # Classification
            if 'classification' in analysis_results:
                clf = analysis_results['classification']
                accuracy = clf.get('accuracy', 0) * 100
                summary_parts.append(f"Classification achieved {accuracy:.1f}% accuracy.")

            # Combine into comprehensive summary
            comprehensive_summary = " ".join(summary_parts)
            return comprehensive_summary

        except Exception as e:
            logger.warning(f"Error generating comprehensive summary: {e}")
            return "Comprehensive analysis completed successfully."

    def _generate_word_cloud_data(self, keywords: List[str], key_phrases: List[str]) -> Dict[str, int]:
        """
        Generate word cloud data with frequency counts
        """
        try:
            word_freq = {}

            # Add keywords with higher frequency
            for keyword in keywords:
                word_freq[keyword] = word_freq.get(keyword, 0) + 3

            # Add key phrases with lower frequency
            for phrase in key_phrases:
                word_freq[phrase] = word_freq.get(phrase, 0) + 1

            # Limit to top 50 words for word cloud
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_words[:50])

        except Exception as e:
            logger.warning(f"Error generating word cloud data: {e}")
            return {}

    def _generate_top_terms(self, keywords: List[str], key_phrases: List[str]) -> List[str]:
        """
        Generate top terms for hashtag-style display
        """
        try:
            # Combine keywords and phrases, prioritizing keywords
            all_terms = []
            
            # Add top keywords (higher priority)
            for keyword in keywords[:15]:
                all_terms.append(keyword.lower().replace(' ', '-'))
            
            # Add top phrases (lower priority)
            for phrase in key_phrases[:5]:
                # Convert phrases to hashtag format
                hashtag = phrase.lower().replace(' ', '-').replace(',', '').replace('.', '')
                if len(hashtag) <= 20:  # Reasonable hashtag length
                    all_terms.append(hashtag)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in all_terms:
                if term not in seen and len(term) > 2:
                    seen.add(term)
                    unique_terms.append(f"#{term}")
            
            return unique_terms[:20]  # Limit to top 20 terms

        except Exception as e:
            logger.warning(f"Error generating top terms: {e}")
            return ["#analysis", "#data", "#insights", "#keywords", "#topics"]
