from fastapi import APIRouter, UploadFile, File
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Read uploaded PDF
    pdf_bytes = await file.read()

    # Open PDF
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

    # Extract text from all pages
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    from app.embedding_model import model
    from app.vector_store import store_chunks

    # Generate embeddings for the chunks
    embeddings = model.encode(chunks).tolist()

    # Store the chunks and their embeddings
    store_chunks(chunks, embeddings)

    # Return information about chunks
    return {
    "filename": file.filename,
    "characters": len(text),
    "total_chunks": len(chunks),
    "stored": True
}