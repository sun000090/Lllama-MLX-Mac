from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from uuid import uuid4
import asyncio
import os

app = FastAPI()

# Store results in memory (use Redis or DB for production)
RESULTS = {}

# OpenAI Client (Replace with your API Key)
OPENAI_API_KEY = "your_openai_api_key"
client = OpenAI(api_key=OPENAI_API_KEY)

async def process_file(file_id: str, content: str):
    """Send file content to OpenAI GPT and store response"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": content[:1000]}]  # Limit input to avoid excessive tokens
        )
        llm_output = response.choices[0].message.content
        RESULTS[file_id] = llm_output
    except Exception as e:
        RESULTS[file_id] = f"Error processing: {str(e)}"

@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a TXT file and starts LLM processing"""
    # Validate file type
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    file_id = str(uuid4())  # Generate unique ID
    content = await file.read()  # Read file content
    
    # Check file size (limit to 1MB for security)
    if len(content) > 1_000_000:
        raise HTTPException(status_code=400, detail="File too large (max 1MB)")

    # Start background processing with OpenAI LLM
    background_tasks.add_task(process_file, file_id, content.decode())

    return {"file_id": file_id, "message": "TXT file received, processing started."}

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    """Retrieves processed result"""
    if file_id in RESULTS:
        return {"file_id": file_id, "result": RESULTS[file_id]}
    return {"file_id": file_id, "message": "Processing not complete or file not found."}
