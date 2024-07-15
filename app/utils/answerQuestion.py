import os
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from ..mysql.savefiles import save_history, get_history_by_filename
from ..utils import prompt
from constants import MODEL_NAME, OPENAI_API_KEY
from logger import setup_logger
from sqlalchemy.orm import Session

logger = setup_logger()

qa_prompt = prompt.prompt
db_directory = './database'


def get_namespace_directory(category, option, namespace):
    if category is None and option is None:
        return os.path.join(db_directory, namespace)
    elif option is None:
        return os.path.join(db_directory, "Categories", category, namespace)
    else:
        return os.path.join(db_directory, "Options", option, namespace)


def load_vectorstore(namespace_directory, embeddings):
    return FAISS.load_local(namespace_directory, embeddings, allow_dangerous_deserialization=True)


def create_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.1
    )


def create_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}
    )


def answer_question(category, option, namespace, question, db: Session):
    try:
        embeddings = OpenAIEmbeddings(
            model=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY
        )

        namespace_directory = get_namespace_directory(category, option, namespace)

        with ThreadPoolExecutor() as executor:
            future_vectorstore = executor.submit(load_vectorstore, namespace_directory, embeddings)
            future_llm = executor.submit(create_llm)

            vectorstore = future_vectorstore.result()
            llm = future_llm.result()

            chain = create_chain(llm, vectorstore)

            result = chain.invoke(question)
            answer = result.get("result", '')

            if option is None:
                save_history(namespace, question, answer, db)

        return result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e