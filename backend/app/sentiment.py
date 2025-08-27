from textblob import TextBlob

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
            
        return {"sentiment": label, "score": round(polarity, 3)}
    except Exception as e:
        return {"sentiment": "neutral", "score": 0.0}
