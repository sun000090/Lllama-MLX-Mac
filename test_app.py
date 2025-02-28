from model import LLMModel
from fastapi import HTTPException, File, UploadFile, FastAPI
import uvicorn

app = FastAPI()

filename = {}

@app.post("/upload")
async def upload(file: UploadFile = File()):
    filename['files'] = file.filename
    return {"message": f"File succesfully uploaded: {file.filename}"}

@app.get("/content")
async def return_content():
    fin = LLMModel.generate_response(filename['files'])
    return {'message': fin}

if __name__ == "__main__":
    uvicorn.run("test_app:app", host="127.0.0.1", port=8000, reload=True)