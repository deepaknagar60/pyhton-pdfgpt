from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ...utils.answerQuestion import answer_question
from ...DTO.dto import QuestionRequest
from ...mysql import database
from ...mysql.savefiles import get_history_by_filename
from logger import setup_logger




logger = setup_logger()
router = APIRouter()
db = database.SessionLocal()


@router.post("/chat")
async def ask_question(request: QuestionRequest):
    try:
        if not request.question or not request.filename:
            raise HTTPException(status_code=400, detail="filename and question parameters are required")
       
        answer = answer_question(request.category,request.filename, request.question)
        
        return JSONResponse(status_code=200, content={"answer": answer})
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Error occured while query..")