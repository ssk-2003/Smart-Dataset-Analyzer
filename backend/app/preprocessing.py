from fastapi import UploadFile
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

async def preprocess_text(file: UploadFile):
    try:
        # Read file content
        text = (await file.read()).decode("utf-8")
        
        # Clean text
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.lower()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        
        cleaned = " ".join(tokens)
        return {"cleaned_text": cleaned}
    except Exception as e:
        return {"cleaned_text": f"Error processing file: {str(e)}"}
