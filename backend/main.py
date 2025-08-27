from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.preprocessing import preprocess_text
from app.topic_model import get_topics
from app.sentiment import analyze_sentiment
from app.summarization import summarize_text
from app.schemas import PreprocessResponse, TopicResponse, SentimentResponse, SummarizeResponse

app = FastAPI(title="Smart Dataset Analyzer API", version="1.0.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Smart Dataset Analyzer API is running!"}

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_api(file: UploadFile = File(...)):
    return await preprocess_text(file)

@app.post("/topic-model", response_model=TopicResponse)
async def topic_model_api(data: PreprocessResponse):
    return get_topics(data.cleaned_text)

@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment_api(data: PreprocessResponse):
    return analyze_sentiment(data.cleaned_text)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_api(data: PreprocessResponse):
    return summarize_text(data.cleaned_text)

@app.post("/analyze-all")
async def analyze_all_api(file: UploadFile = File(...)):
    """Complete analysis pipeline in one endpoint"""
    # Preprocess
    preprocess_result = await preprocess_text(file)
    cleaned_text = preprocess_result["cleaned_text"]
    
    # Run all analyses
    topics = get_topics(cleaned_text)
    sentiment = analyze_sentiment(cleaned_text)
    summary = summarize_text(cleaned_text)
    
    return {
        "preprocessing": preprocess_result,
        "topics": topics,
        "sentiment": sentiment,
        "summary": summary
    }
