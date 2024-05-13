import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
from app.mysql import database
from app.controllers import controller

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

database.Base.metadata.create_all(bind=database.engine)   # to create tables automatically if not exist in db

app.include_router(controller.router)



if __name__ == "__main__":
    uvicorn.run("main:app",port=8001, reload=True) 