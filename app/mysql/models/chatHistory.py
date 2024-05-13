from sqlalchemy import Boolean, Column,Integer, String
from sqlalchemy.orm import relationship
from ...mysql.database import Base 


class ChatHistory(Base):
    __tablename__ = "ChatHistory"
    id = Column(Integer,primary_key=True,index=True)
    filename = Column(String(255),index=True)
    question = Column(String(2000), index=True)  
    answer = Column(String(3000), index=True)  
   