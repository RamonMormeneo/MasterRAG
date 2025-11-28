# importamos la API KEY

import os                           # libreria del sistema operativo
from dotenv import load_dotenv      # carga variables de entorno 


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#probando gpt4

from langchain_openai.chat_models import ChatOpenAI   

modelo = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-4-turbo')

respuesta = modelo.invoke("Who is Apple's CEO?")

respuesta.content

os.listdir('../pdfs')

from langchain_community.document_loaders import PyPDFDirectoryLoader

# loads PDF file page by page

loader = PyPDFDirectoryLoader('pdfs/')

paginas = loader.load()

len(paginas)

#dividirlo por chunks

chunks = loader.load_and_split()

#embreding

from langchain_openai.embeddings import OpenAIEmbeddings


vectorizador = OpenAIEmbeddings()

from langchain_chroma import Chroma

chroma_db = Chroma.from_documents(chunks, vectorizador, persist_directory='../chroma_db')

#carga desde chroma

consulta = 'What can you tell me about foreign exchange contracts?'

chroma_db = Chroma(persist_directory='../chroma_db', embedding_function=vectorizador)

docs = chroma_db.similarity_search(consulta, k=10)

len(docs)

recuperador = chroma_db.as_retriever(search_type='mmr', search_kwargs={'k': 15, 'lambda_mult': 0.25})

#prompt

from langchain.prompts import ChatPromptTemplate

template = '''
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            '''


prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

cadena = {'context': recuperador, 'question': RunnablePassthrough()} | prompt | modelo | parser


respuesta = cadena.invoke(consulta)



consulta = 'What can you tell me about foreign exchange contracts?'

cadena.invoke(consulta)