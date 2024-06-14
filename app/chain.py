from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os, getpass
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt=ChatPromptTemplate.from_template(template)

llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=1)

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

model_name = 'text-embedding-ada-002'  
embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=OPENAI_API_KEY
)

vectorstore=PineconeVectorStore(index_name="documents",embedding=embeddings,namespace='schalk-burger')
retriever = vectorstore.as_retriever()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Where is Schalk now?")