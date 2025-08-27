from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize

def summarize_text(text):
    try:
        # Simple extractive summarization using sentence ranking
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return {"summary": text}
        
        # Score sentences by word frequency (simple approach)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words_in_sentence = sentence.lower().split()
            score = 0
            word_count = 0
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x, reverse=True)
        summary_length = min(2, len(top_sentences))
        summary = " ".join([sent for sent in top_sentences[:summary_length]])
        
        return {"summary": summary if summary else text[:200] + "..."}
    except Exception as e:
        return {"summary": text[:200] + "..." if len(text) > 200 else text}
