from fastapi import APIRouter, HTTPException,Depends
import os
from ...mysql import database
from ...mysql.models import files,categories
from sqlalchemy.orm import Session
from logger import setup_logger



logger = setup_logger()
router = APIRouter()


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_directory = './database'

@router.get("/files")
async def get_filenames(db: Session = Depends(get_db)):
    try: 
        allfiles = db.query(files.Files).all()
        file_dict = {file.filename: file.createdAt for file in allfiles}

        
        filename = [name for name in os.listdir(db_directory) 
                      if os.path.isdir(os.path.join(db_directory, name)) and name != "Categories"]

        filename.sort(key=lambda x: file_dict.get(x, ''), reverse=True)
        
        return {"files": filename}
       
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Error occured while fetching the files..")
    
    

db_directory_category = 'database/Categories'

@router.get("/category")
async def get_categories(db: Session = Depends(get_db)):
    try: 
        files = db.query(categories.Categories).all()
        file_dict = {file.filename: file.createdAt for file in files}

        
        category = [name for name in os.listdir(db_directory_category)
                       if os.path.isdir(os.path.join(db_directory_category, name))]
        
        category.sort(key=lambda x: file_dict.get(x, ''), reverse=True)
        return {"categories": category}
       
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Error occured while fetching the categories..")    