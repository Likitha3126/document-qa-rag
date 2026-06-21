from app.upload import router as upload_router
from fastapi import FastAPI
from pydantic import BaseModel, Field
import ollama
from app.vector_store import retrieve_chunks
from app.embedding_model import model
from app.document_state import current_document
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Document Q&A API",
    description="RAG-based Document Question Answering using Qwen3, ChromaDB and FastAPI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "Backend Running"}

@app.get("/health")
def health():
    return {"status": "ok"}
app.include_router(upload_router)

@app.get("/document-info")
def document_info():
    return current_document


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500
    )


@app.get("/")
def home():
    return {"message": "Backend Running"}


@app.post("/chat")
def chat(request: ChatRequest):

    query_embedding = model.encode(
        request.question
    ).tolist()

    retrieval_result = retrieve_chunks(query_embedding)

    retrieved_chunks = retrieval_result["documents"]
    retrieved_metadata = retrieval_result["metadata"]

    context = "\n\n".join(retrieved_chunks)


    prompt = f"""
You are a document question-answering assistant.

Answer using ONLY the provided context.

If the answer is not present in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{request.question}

Answer:
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
    "sources": retrieved_metadata
}