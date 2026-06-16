from app.upload import router as upload_router
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from app.vector_store import retrieve_chunks
from app.embedding_model import model

app = FastAPI(title="Document Q&A API")
app.include_router(upload_router)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Backend Running"}


@app.post("/chat")
def chat(request: ChatRequest):

    query_embedding = model.encode(
        request.question
    ).tolist()

    retrieved_chunks = retrieve_chunks(
        query_embedding
    )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Answer ONLY using the provided context.

    Context:
    {context}

    Question:
    {request.question}
    """

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": "You are a document question-answering assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "answer": response["message"]["content"],
        "sources": retrieved_chunks
    }