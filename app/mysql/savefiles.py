
from fastapi import FastAPI,HTTPException, Depends
from datetime import datetime
from sqlalchemy.orm import Session
from ..mysql.models import files,categories,chatHistory 
from datetime import datetime
from logger import setup_logger



logger = setup_logger()

current_date_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    

def insert_file_name(filename: str, db: Session):
    try:
        new_item = files.Files(filename=filename, createdAt=current_date_time)
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        
        logger.info("file name saved in mysql database...............")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    
    


def insert_category(filename: str, db: Session):
    try:
        new_item = categories.Categories(filename=filename, createdAt=current_date_time)
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        
        logger.info("category saved in mysql database...............")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")    
    
    

def save_history(filename: str, question: str, answer:str, db:Session):
    try:
        History = chatHistory.ChatHistory(filename=filename, question=question, answer=answer)
        db.add(History)
        db.commit()
        db.refresh(History)
        
        logger.info("chat history saved in mysql databse.........")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")            
    
    
def get_history_by_filename(filename: str, db: Session):
    try:
        chat_history = db.query(models.ChatHistory).filter(models.ChatHistory.filename == filename).all()
        if chat_history:
            return [{row.question, row.answer} for row in chat_history]
        else:
            return []  
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")    