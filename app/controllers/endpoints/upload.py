import os
from fastapi import APIRouter, UploadFile, File, Form
from ...utils.store_data import run
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from logger import setup_logger




logger = setup_logger()
router = APIRouter()




@router.post("/upload")
async def upload_file(category: str = Form(default=None),file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse(status_code=400, content={'error': 'Only PDF files are allowed'})
        
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        
        file_content = await file.read()
        
        with open(file_path, 'wb+') as destination:
            destination.write(file_content)
        
        run(file.filename, category)
        
        os.remove(file_path)
        
        return JSONResponse(status_code=200, content={'message': 'File uploaded successfully', 'file_path': file_path})
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(status_code=500, error='Error occured while uploading file..')
