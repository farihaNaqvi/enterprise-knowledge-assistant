from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.ingestion import IngestionService
from app.services.vector_store import get_vector_store
from app.services.rag_chain import create_rag_assistant
import shutil
import os

app = FastAPI(title="Enterprise GenAI Assistant")
ingest_service = IngestionService()

# Global state to keep the assistant active after upload
# In a multi-user app, you'd use a database, but for a portfolio, this works perfectly.
rag_chain = None


@app.get("/")
async def root():
    return {"message": "Enterprise Assistant API is online."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Save file locally (Standard Enterprise pattern: landing zone)
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Ingest and Chunk
        chunks = await ingest_service.process_pdf(temp_path)

        # 3. Create Vector Store & Assistant
        vector_db = get_vector_store(chunks)

        global rag_chain
        rag_chain = create_rag_assistant(vector_db)

        return {
            "status": "Success",
            "message": f"Document '{file.filename}' indexed. You can now ask questions."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 4. Clean up the temp file even if it fails
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/query")
async def ask_question(q: str):
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No document uploaded. Use /upload first.")

    # Trigger the RAG chain
    response = rag_chain.invoke({"input": q})

    # 'context' contains the chunks the AI actually read
    sources = [doc.metadata.get("source", "Unknown") for doc in response["context"]]

    return {
        "answer": response["answer"],
        "citations": list(set(sources))  # Unique sources only
    }