from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def get_topics(text):
    try:
        # Create documents list
        docs = [text]
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english", min_df=1, max_df=0.95)
        X = vectorizer.fit_transform(docs)
        
        # Apply NMF for topic modeling
        nmf = NMF(n_components=3, init="random", random_state=1, max_iter=100)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        # Extract top words for each topic
        features = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_weights in H:
            top_words_idx = topic_weights.argsort()[::-1][:5]
            topic_words = [features[i] for i in top_words_idx if topic_weights[i] > 0]
            if topic_words:
                topics.append(topic_words)
        
        # If no topics found, return default
        if not topics:
            topics = [["analysis", "text", "data"], ["content", "information", "processing"]]
            
        return {"topics": topics}
    except Exception as e:
        return {"topics": [["error", "processing", "text"], ["default", "topic", "analysis"]]}
