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

    # Return information about chunks
    return {
        "filename": file.filename,
        "characters": len(text),
        "total_chunks": len(chunks),
        "first_chunk": chunks[0] if chunks else ""
    }