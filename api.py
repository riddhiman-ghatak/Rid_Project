
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from agents import SearchAgent, QAAgent, FutureWorksAgent

app = FastAPI()


search_agent = SearchAgent()
qa_agent = QAAgent()
future_works_agent = FutureWorksAgent()


class TopicRequest(BaseModel):
    topic: str

class QARequest(BaseModel):
    question: str
    context: str

@app.post("/search")
async def search_papers(request: TopicRequest):
    try:
        papers = search_agent.search_papers(request.topic)
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def answer_question(request: QARequest):
    try:
        answer = qa_agent.answer_question(request.question, request.context)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/future-directions")
async def get_future_directions(request: TopicRequest):
    try:
        papers = search_agent.search_papers(request.topic)
        directions = future_works_agent.generate_future_directions(papers)
        return {"directions": directions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)