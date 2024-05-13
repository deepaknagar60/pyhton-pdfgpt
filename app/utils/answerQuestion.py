import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from ..mysql.savefiles import save_history,get_history_by_filename
from ..mysql import database
from ..utils import prompt
from constants import MODEL_NAME,OPENAI_API_KEY
from logger import setup_logger



logger = setup_logger()

qa_prompt = prompt.prompt

db = database.SessionLocal()
db_directory = './database'

 
def answer_question(category, namespace , question):
    try:
        
        embeddings = OpenAIEmbeddings(
            model=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY
        )
         
        
         
        if category is None:
            namespace_directory = os.path.join(db_directory, namespace)
        
        else:
            namespace_directory = os.path.join(db_directory,category, namespace)
            
        
        vectorstore = FAISS.load_local(namespace_directory, embeddings,allow_dangerous_deserialization=True)
        
        # history = get_history_by_filename(namespace,db)
        # print(history)
        # query_result = vectorstore.similarity_search(query=question, k=3)
        
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-3.5-turbo',
            temperature=0.1
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": qa_prompt}, 
            # return_source_documents=True
        )

        result = chain.invoke(question)
        answer = result.get( "result",'')
        
        save_history(namespace, question, answer,db)
    
        return result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
