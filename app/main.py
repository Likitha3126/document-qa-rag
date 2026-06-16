from app.upload import router as upload_router
from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI(title="Document Q&A API")
app.include_router(upload_router)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Backend Running"}


@app.post("/chat")
def chat(request: ChatRequest):

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant. Give short direct answers."
            },
            {
                "role": "user",
                "content": request.question
            }
        ]
    )

    return {
        "answer": response["message"]["content"]
    }
