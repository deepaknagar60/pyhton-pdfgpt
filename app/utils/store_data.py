import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from ..mysql.savefiles import insert_file_name, insert_category
from ..mysql import models, database
from constants import MODEL_NAME,OPENAI_API_KEY
from logger import setup_logger



logger = setup_logger()
db = database.SessionLocal()
directory = './uploads'
db_directory = './database'


def run(namespace , category=None):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=4000,
    chunk_overlap=0,
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key= OPENAI_API_KEY
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
     
    if category is None:
        namespace_directory = os.path.join(db_directory, namespace)
    else:
        namespace_directory = os.path.join(db_directory,"Categories", category)
     
        
    index_file_path = os.path.join(namespace_directory, 'index.faiss')
    if os.path.exists(index_file_path):
        vectorstoreExisting = FAISS.load_local(namespace_directory, embeddings, allow_dangerous_deserialization=True)
        vectorstoreExisting.add_documents(docs)
        vectorstoreExisting.save_local(namespace_directory) 
        
    else: 
        vectorstore.save_local(namespace_directory)
        if category is None:
            insert_file_name(namespace,db)
        
        else:
            insert_category(category,db)        
    
    logger.info("Data stored in vector database......")
  
    




