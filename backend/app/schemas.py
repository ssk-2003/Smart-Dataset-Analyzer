from pydantic import BaseModel
from typing import List

class PreprocessResponse(BaseModel):
    cleaned_text: str

class TopicResponse(BaseModel):
    topics: List[List[str]]

class SentimentResponse(BaseModel):
    sentiment: str
    score: float

class SummarizeResponse(BaseModel):
    summary: str
