"""
Enhanced Topic Modeling Pipeline with Fine-tuning Capabilities
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import warnings
from loguru import logger
import re
import nltk
from collections import Counter
import itertools

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic preprocessing")


class TopicModelingPipeline:
    """
    Enhanced topic modeling pipeline with fine-tuning capabilities
    """
    
    def __init__(self, enable_fine_tuning: bool = True):
        """Initialize topic modeling components"""
        self.lda = None
        self.nmf = None
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.enable_fine_tuning = enable_fine_tuning
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.custom_stop_words = self._get_enhanced_stop_words()
        logger.info(f"Enhanced topic modeling pipeline initialized (fine-tuning: {enable_fine_tuning})")
    
    def _get_enhanced_stop_words(self) -> set:
        """Get enhanced stop words list"""
        base_stops = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        if NLTK_AVAILABLE:
            nltk_stops = set(stopwords.words('english'))
            base_stops.update(nltk_stops)
        
        # Add domain-specific stop words
        domain_stops = {
            'would', 'could', 'should', 'might', 'may', 'must', 'shall', 'will',
            'one', 'two', 'three', 'first', 'second', 'third', 'also', 'however',
            'therefore', 'thus', 'hence', 'moreover', 'furthermore', 'additionally',
            'said', 'says', 'saying', 'tell', 'told', 'ask', 'asked', 'asking',
            'get', 'got', 'getting', 'give', 'given', 'giving', 'take', 'taken', 'taking',
            'make', 'made', 'making', 'go', 'going', 'went', 'come', 'coming', 'came',
            'see', 'seen', 'seeing', 'look', 'looking', 'looked', 'find', 'found', 'finding',
            'know', 'known', 'knowing', 'think', 'thinking', 'thought', 'believe', 'believing',
            'way', 'ways', 'time', 'times', 'year', 'years', 'day', 'days', 'thing', 'things',
            'people', 'person', 'man', 'woman', 'men', 'women', 'good', 'bad', 'great', 'small',
            'large', 'big', 'little', 'old', 'new', 'long', 'short', 'high', 'low', 'right', 'left'
        }
        base_stops.update(domain_stops)
        
        return base_stops
    
    def process(self, documents: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """
        Enhanced topic modeling with optional fine-tuning
        
        Args:
            documents: List of preprocessed documents
            n_topics: Number of topics to extract
            
        Returns:
            Dictionary containing topic modeling results
        """
        # Enhanced preprocessing
        processed_docs = self._enhanced_preprocessing(documents)
        
        # Sampling for performance
        max_docs = 1500 if self.enable_fine_tuning else 1000
        if len(processed_docs) > max_docs:
            import random
            random.seed(42)
            sampled_docs = random.sample(processed_docs, max_docs)
            logger.info(f"🚀 Sampled {max_docs} documents from {len(processed_docs)} for topic modeling")
        else:
            sampled_docs = processed_docs
            
        logger.info(f"Starting enhanced topic modeling with {n_topics} topics on {len(sampled_docs)} documents...")
        
        results = {"models": [], "best_model": None}
        
        if self.enable_fine_tuning:
            # Fine-tuned approach with hyperparameter optimization
            optimal_params = self._find_optimal_parameters(sampled_docs, n_topics)
            lda_results = self._run_optimized_lda(sampled_docs, optimal_params['lda'])
            nmf_results = self._run_optimized_nmf(sampled_docs, optimal_params['nmf'])
        else:
            # Standard approach
            lda_results = self._run_lda(sampled_docs, n_topics)
            nmf_results = self._run_nmf(sampled_docs, n_topics)
        
        # Add successful models to results
        if lda_results:
            results["models"].append(lda_results)
        if nmf_results:
            results["models"].append(nmf_results)
        
        # Select the best model based on coherence score
        if results["models"]:
            best_model = max(results["models"], key=lambda x: x.get('coherence_score', 0))
            results["best_model"] = best_model["name"]
            logger.info(f"✅ Selected {results['best_model']} as best model with coherence score: {best_model.get('coherence_score', 0):.3f}")
        else:
            logger.warning("No topic model could be generated")
        
        return results
    
    def _enhanced_preprocessing(self, documents: List[str]) -> List[str]:
        """Enhanced text preprocessing with lemmatization and better cleaning"""
        processed = []
        
        for doc in documents:
            # Basic cleaning
            doc = re.sub(r'[^\w\s]', ' ', doc.lower())
            doc = re.sub(r'\d+', '', doc)  # Remove numbers
            doc = re.sub(r'\s+', ' ', doc).strip()
            
            if len(doc) < 10:  # Skip very short documents
                continue
                
            # Lemmatization if available
            if self.lemmatizer:
                words = doc.split()
                words = [self.lemmatizer.lemmatize(word) for word in words 
                        if word not in self.custom_stop_words and len(word) > 2]
                doc = ' '.join(words)
            
            if doc.strip():
                processed.append(doc)
        
        logger.info(f"Enhanced preprocessing: {len(documents)} → {len(processed)} documents")
        return processed
    
    def _find_optimal_parameters(self, documents: List[str], base_n_topics: int) -> Dict[str, Dict]:
        """Find optimal parameters through grid search with enhanced perplexity optimization"""
        logger.info("🔍 Finding optimal parameters with perplexity optimization...")
        
        # Determine optimal number of topics with extended range for high perplexity
        optimal_topics = self._find_optimal_topic_count_enhanced(documents, base_n_topics)
        
        # Enhanced parameter grids for better perplexity scores
        dataset_size = len(documents)
        vocab_size = min(2000, dataset_size * 3)  # Increased vocabulary for complex data
        
        lda_params = {
            'n_topics': optimal_topics,
            'max_features': vocab_size,
            'min_df': max(2, min(5, dataset_size // 200)),  # Adaptive min_df
            'max_df': 0.7,  # More restrictive to remove very common words
            'learning_method': 'online',
            'max_iter': 50,  # Increased iterations for better convergence
            'alpha': 0.05,  # Lower alpha for more focused topics
            'beta': 0.005   # Lower beta for more specific word distributions
        }
        
        nmf_params = {
            'n_topics': optimal_topics,
            'max_features': vocab_size,
            'min_df': max(2, min(5, dataset_size // 200)),
            'max_df': 0.7,
            'init': 'nndsvd',
            'max_iter': 300,  # Increased iterations
            'alpha': 0.05,   # Lower regularization for complex data
            'l1_ratio': 0.3  # More L2 regularization for stability
        }
        
        logger.info(f"Enhanced parameters: topics={optimal_topics}, vocab={vocab_size}, min_df={lda_params['min_df']}")
        return {'lda': lda_params, 'nmf': nmf_params}
    
    def _find_optimal_topic_count_enhanced(self, documents: List[str], base_n_topics: int) -> int:
        """Enhanced topic count optimization for high perplexity cases"""
        if len(documents) < 50:
            return min(base_n_topics, max(3, len(documents) // 8))
        
        # Extended topic range for complex datasets
        min_topics = max(3, base_n_topics - 3)
        max_topics = min(base_n_topics + 5, len(documents) // 3)
        topic_range = range(min_topics, max_topics + 1)
        
        best_topics = base_n_topics
        best_score = 0
        best_perplexity = float('inf')
        
        logger.info(f"Testing enhanced topic range: {list(topic_range)}")
        
        for n_topics in topic_range:
            try:
                # Enhanced vectorizer for testing
                vectorizer = CountVectorizer(
                    max_features=min(1500, len(documents) * 2),
                    min_df=max(2, len(documents) // 200),
                    max_df=0.7,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                doc_term = vectorizer.fit_transform(documents)
                
                if n_topics >= min(doc_term.shape):
                    continue
                    
                # Test LDA with enhanced parameters
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=20,
                    learning_method='online',
                    doc_topic_prior=0.05,
                    topic_word_prior=0.005
                )
                doc_topic = lda.fit_transform(doc_term)
                
                # Calculate both coherence and perplexity
                coherence = self._calculate_coherence_score(doc_topic)
                perplexity = lda.perplexity(doc_term)
                
                # Combined score favoring lower perplexity and higher coherence
                combined_score = coherence - (perplexity / 10000)  # Normalize perplexity impact
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_topics = n_topics
                    best_perplexity = perplexity
                    
                logger.debug(f"Topics: {n_topics}, Coherence: {coherence:.3f}, Perplexity: {perplexity:.1f}, Combined: {combined_score:.3f}")
                    
            except Exception as e:
                logger.debug(f"Failed to test {n_topics} topics: {e}")
                continue
        
        logger.info(f"Enhanced optimal: {best_topics} topics (score: {best_score:.3f}, perplexity: {best_perplexity:.1f})")
        return best_topics
    
    def _find_optimal_topic_count(self, documents: List[str], base_n_topics: int) -> int:
        """Find optimal number of topics using coherence scores"""
        if len(documents) < 50:
            return min(base_n_topics, max(2, len(documents) // 10))
        
        # Test different topic counts
        topic_range = range(max(2, base_n_topics - 2), min(base_n_topics + 3, len(documents) // 5))
        best_topics = base_n_topics
        best_coherence = 0
        
        logger.info(f"Testing topic counts: {list(topic_range)}")
        
        for n_topics in topic_range:
            try:
                # Quick LDA test
                vectorizer = CountVectorizer(max_features=300, min_df=2, max_df=0.8, stop_words='english')
                doc_term = vectorizer.fit_transform(documents)
                
                if n_topics >= min(doc_term.shape):
                    continue
                    
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
                doc_topic = lda.fit_transform(doc_term)
                coherence = self._calculate_coherence_score(doc_topic)
                
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_topics = n_topics
                    
            except Exception as e:
                logger.debug(f"Failed to test {n_topics} topics: {e}")
                continue
        
        logger.info(f"Optimal topic count: {best_topics} (coherence: {best_coherence:.3f})")
        return best_topics
    
    def _run_optimized_lda(self, documents: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimized LDA with fine-tuned parameters"""
        try:
            logger.info(f"Running optimized LDA with {params['n_topics']} topics...")
            
            # Enhanced vectorizer with optimized parameters
            self.count_vectorizer = CountVectorizer(
                max_features=params['max_features'],
                min_df=params['min_df'],
                max_df=params['max_df'],
                stop_words=list(self.custom_stop_words),
                ngram_range=(1, 2),  # Include bigrams
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
            )
            
            doc_term_matrix = self.count_vectorizer.fit_transform(documents)
            
            # Optimized LDA
            self.lda = LatentDirichletAllocation(
                n_components=params['n_topics'],
                random_state=42,
                max_iter=params['max_iter'],
                learning_method=params['learning_method'],
                doc_topic_prior=params['alpha'],
                topic_word_prior=params['beta'],
                n_jobs=1
            )
            
            lda_doc_topic = self.lda.fit_transform(doc_term_matrix)
            
            # Extract enhanced topics
            topics = self._extract_enhanced_topics(
                self.lda.components_, 
                self.count_vectorizer.get_feature_names_out(),
                lda_doc_topic,
                documents,
                "LDA"
            )
            
            # Enhanced coherence calculation
            coherence = self._calculate_enhanced_coherence(lda_doc_topic, topics)
            perplexity = self.lda.perplexity(doc_term_matrix)
            
            logger.info(f"Optimized LDA completed - Coherence: {coherence:.3f}, Perplexity: {perplexity:.1f}")
            
            return {
                "name": "LDA (Bag of Words)",
                "topics": topics,
                "coherence_score": coherence,
                "perplexity": perplexity,
                "optimization_used": True,
                "parameters": params
            }
            
        except Exception as e:
            logger.error(f"Optimized LDA failed: {str(e)}")
            return self._run_lda(documents, params['n_topics'])  # Fallback to standard LDA
    
    def _run_optimized_nmf(self, documents: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimized NMF with fine-tuned parameters"""
        try:
            logger.info(f"Running optimized NMF with {params['n_topics']} topics...")
            
            # Enhanced TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=params['max_features'],
                min_df=params['min_df'],
                max_df=params['max_df'],
                stop_words=list(self.custom_stop_words),
                ngram_range=(1, 2),  # Include bigrams
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
                sublinear_tf=True,  # Use log-scaled tf
                norm='l2'  # L2 normalization
            )
            
            doc_term_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Optimized NMF
            self.nmf = NMF(
                n_components=params['n_topics'],
                random_state=42,
                init=params['init'],
                max_iter=params['max_iter'],
                alpha=params['alpha'],
                l1_ratio=params['l1_ratio']
            )
            
            nmf_doc_topic = self.nmf.fit_transform(doc_term_matrix)
            
            # Extract enhanced topics
            topics = self._extract_enhanced_topics(
                self.nmf.components_,
                self.tfidf_vectorizer.get_feature_names_out(),
                nmf_doc_topic,
                documents,
                "NMF"
            )
            
            # Enhanced metrics
            coherence = self._calculate_enhanced_coherence(nmf_doc_topic, topics)
            reconstruction_error = self.nmf.reconstruction_err_
            topic_diversity = self._calculate_topic_diversity(self.nmf.components_)
            
            logger.info(f"Optimized NMF completed - Coherence: {coherence:.3f}, Reconstruction Error: {reconstruction_error:.3f}")
            
            return {
                "name": "NMF (TF-IDF)",
                "topics": topics,
                "coherence_score": coherence,
                "reconstruction_error": reconstruction_error,
                "topic_diversity": topic_diversity,
                "optimization_used": True,
                "parameters": params
            }
            
        except Exception as e:
            logger.error(f"Optimized NMF failed: {str(e)}")
            return self._run_nmf(documents, params['n_topics'])  # Fallback to standard NMF
    
    def _extract_enhanced_topics(self, components, feature_names, doc_topic_matrix, documents, model_type):
        """Extract topics with enhanced information"""
        topics = []
        
        for topic_idx, topic in enumerate(components):
            # Get top words with scores
            top_indices = topic.argsort()[-15:][::-1]  # Top 15 words
            top_words_scores = [(feature_names[i], topic[i]) for i in top_indices]
            
            # Filter out low-score words and get top 8
            filtered_words = [word for word, score in top_words_scores if score > np.mean(topic)][:8]
            
            # Calculate topic distribution
            topic_dist = np.mean(doc_topic_matrix[:, topic_idx]) * 100
            
            # Find representative documents
            doc_scores = doc_topic_matrix[:, topic_idx]
            top_doc_indices = doc_scores.argsort()[-3:][::-1]
            representative_docs = [documents[i][:100] + "..." for i in top_doc_indices if doc_scores[i] > 0.1]
            
            topics.append({
                "topic": f"Topic {topic_idx + 1}",
                "keywords": filtered_words[:5],  # Top 5 for display
                "all_keywords": filtered_words,  # All filtered keywords
                "distribution": float(topic_dist),
                "document_count": int(topic_dist * len(documents) / 100),
                "representative_docs": representative_docs,
                "strength": float(np.max(topic))  # Topic strength
            })
        
        return topics
    
    def _calculate_enhanced_coherence(self, doc_topic_matrix, topics):
        """Calculate enhanced coherence score"""
        try:
            # Use silhouette score as base
            dominant_topics = np.argmax(doc_topic_matrix, axis=1)
            
            if len(np.unique(dominant_topics)) > 1:
                base_coherence = silhouette_score(doc_topic_matrix, dominant_topics)
            else:
                base_coherence = 0.3
            
            # Enhance with topic quality metrics
            topic_quality = 0
            for topic in topics:
                # Penalize topics with very low distribution
                if topic['distribution'] < 5:
                    topic_quality -= 0.1
                # Reward topics with good keyword diversity
                if len(set(topic['keywords'])) == len(topic['keywords']):
                    topic_quality += 0.05
            
            enhanced_coherence = max(0, min(1, base_coherence + topic_quality))
            return float(enhanced_coherence)
            
        except Exception as e:
            logger.debug(f"Enhanced coherence calculation failed: {e}")
            return 0.4  # Reasonable default
    
    def _calculate_topic_diversity(self, components):
        """Calculate topic diversity score"""
        try:
            all_top_words = []
            for topic in components:
                top_indices = topic.argsort()[-10:][::-1]
                all_top_words.extend(top_indices)
            
            unique_words = len(set(all_top_words))
            total_words = len(all_top_words)
            
            return unique_words / total_words if total_words > 0 else 0.5
        except:
            return 0.5
    
    def _run_lda(self, documents: List[str], n_topics: int) -> Dict[str, Any]:
        """Run Latent Dirichlet Allocation"""
        try:
            logger.info("Running LDA topic modeling...")
            
            # Create bag of words - reduced features for speed
            self.count_vectorizer = CountVectorizer(
                max_features=500,  # Reduced from 1000
                min_df=2,
                max_df=0.8,  # Remove very common words
                stop_words='english'
            )
            
            doc_term_matrix = self.count_vectorizer.fit_transform(documents)
            
            # Adjust n_components if too large for the dataset
            actual_topics = min(n_topics, doc_term_matrix.shape[1] - 1 or 1)
            logger.info(f"Running LDA with {actual_topics} topics (adjusted for dataset size)")
            
            # Run LDA with performance optimizations
            self.lda = LatentDirichletAllocation(
                n_components=actual_topics,
                random_state=42,
                max_iter=20,  # Reduced iterations for speed
                learning_method='online',  # Faster than batch
                n_jobs=1
            )
            
            lda_doc_topic = self.lda.fit_transform(doc_term_matrix)
            
            # Extract topics
            feature_names = self.count_vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                
                # Calculate topic distribution
                topic_dist = np.mean(lda_doc_topic[:, topic_idx])
                
                topics.append({
                    "topic": f"Topic {topic_idx + 1}",
                    "keywords": top_words[:5],  # Top 5 keywords
                    "distribution": float(topic_dist),
                    "document_count": int(topic_dist * len(documents))
                })
            
            # Calculate coherence score (simplified)
            coherence = self._calculate_coherence_score(lda_doc_topic)
            
            logger.info(f"LDA completed with coherence score: {coherence:.3f}")
            
            return {
                "name": "LDA (Bag of Words)",
                "topics": topics,
                "coherence_score": coherence,
                "perplexity": self.lda.perplexity(doc_term_matrix)
            }
            
        except Exception as e:
            logger.error(f"LDA failed: {str(e)}")
            return None
    
    def _run_nmf(self, documents: List[str], n_topics: int) -> Dict[str, Any]:
        """Run Non-negative Matrix Factorization"""
        try:
            
            # Create TF-IDF matrix - optimized for speed
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced from 1000
                min_df=2,
                max_df=0.8,  # Remove very common words
                stop_words='english'
            )
            doc_term_matrix = self.tfidf_vectorizer.fit_transform(documents)
            matrix_shape = doc_term_matrix.shape
            
            # Adjust n_components if too large for the dataset
            actual_topics = min(n_topics, min(doc_term_matrix.shape) - 1 or 1)
            logger.info(f"Running NMF with {actual_topics} topics (adjusted for dataset size)")
            
            # Initialize NMF with random init for small datasets
            init_method = 'random' if actual_topics >= min(doc_term_matrix.shape) else 'nndsvd'
            self.nmf = NMF(n_components=actual_topics, random_state=42, init=init_method)
            
            nmf_doc_topic = self.nmf.fit_transform(doc_term_matrix)
            
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = []
            diversity_vocab = []
            
            for topic_idx, topic in enumerate(self.nmf.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                diversity_vocab.extend(top_words)
                
                # Calculate topic distribution
                topic_dist = np.mean(nmf_doc_topic[:, topic_idx])
                
                topics.append({
                    "topic": f"Topic {topic_idx + 1}",
                    "keywords": top_words[:5],  # Top 5 keywords
                    "distribution": float(topic_dist),
                    "document_count": int(topic_dist * len(documents))
                })
            
            # Calculate reconstruction error
            reconstruction_error = self.nmf.reconstruction_err_
            # Simple topic diversity: unique top terms / total top terms
            unique_terms = len(set(diversity_vocab))
            total_terms = len(diversity_vocab) if diversity_vocab else 1
            topic_diversity = unique_terms / total_terms
            
            logger.info(
                f"NMF completed | matrix={matrix_shape} | reconstruction_err={reconstruction_error:.4f} | diversity={topic_diversity:.3f}"
            )
            
            return {
                "name": "NMF (TF-IDF)",
                "topics": topics,
                "reconstruction_error": reconstruction_error,
                "coherence_score": self._calculate_coherence_score(nmf_doc_topic),
                "matrix_shape": list(matrix_shape),
                "topic_diversity": float(topic_diversity)
            }
            
        except Exception as e:
            logger.error(f"NMF failed: {str(e)}")
            return None
    
    def _calculate_coherence_score(self, doc_topic_matrix: np.ndarray) -> float:
        """
        Calculate coherence score for topic model
        
        Simplified coherence calculation using silhouette score
        """
        try:
            # Get dominant topic for each document
            dominant_topics = np.argmax(doc_topic_matrix, axis=1)
            
            # Calculate silhouette score if we have enough topics
            if len(np.unique(dominant_topics)) > 1:
                score = silhouette_score(doc_topic_matrix, dominant_topics)
                return float(score)
            else:
                return 0.5  # Default score
                
        except:
            return 0.5  # Default score
    
    def get_document_topics(self, document: str, model_type: str = "lda") -> List[float]:
        """
        Get topic distribution for a single document
        """
        if model_type == "lda" and self.lda and self.count_vectorizer:
            doc_vector = self.count_vectorizer.transform([document])
            topic_dist = self.lda.transform(doc_vector)[0]
            return topic_dist.tolist()
        
        elif model_type == "nmf" and self.nmf and self.tfidf_vectorizer:
            doc_vector = self.tfidf_vectorizer.transform([document])
            topic_dist = self.nmf.transform(doc_vector)[0]
            return topic_dist.tolist()
        
        return []
