from sqlalchemy import Boolean, Column,Integer, String
from sqlalchemy.orm import relationship
from ...mysql.database import Base 




class Files(Base):
    __tablename__ = "Files"
    id = Column(Integer,primary_key=True,index=True)
    filename = Column(String(255),index=True)
    createdAt = Column(String(255), index=True)