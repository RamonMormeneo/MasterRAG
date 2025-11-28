# importamos la API KEY

import os                           # libreria del sistema operativo
from dotenv import load_dotenv      # carga variables de entorno 


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langchain_openai.chat_models import ChatOpenAI   

modelo = ChatOpenAI(model='gpt-4-turbo')

respuesta = modelo.invoke('What is the Suez Canal?')

respuesta.content
os.listdir('/pdfs')

from langchain_community.document_loaders import PyPDFDirectoryLoader

# cargar archivo pdf 

loader = PyPDFDirectoryLoader('../pdfs/')

paginas = loader.load()

# nº de paginas en el pdf

len(paginas)

# modelo embedding de OpenAI

from langchain_openai.embeddings import OpenAIEmbeddings

vectorizador = OpenAIEmbeddings()

# guardando vectores en ChromaDB

from langchain_community.vectorstores import Chroma

chroma_db = Chroma.from_documents(paginas, vectorizador, persist_directory='/chroma_db')

# objecto para recuperar 2 paginasdesde la base de datos

recuperador = chroma_db.as_retriever(search_type='mmr', search_kwargs={'k': 2, 'lambda_mult': 0.25})

from langchain.prompts import ChatPromptTemplate

template = '''
            Given the context below and the question, 
            please generate a header and 10 bullet points.
            List with numbers the bullet points.
            Summarize each bullet point in 40 words.
            
            Put a line separator after `:` symbol.

            Context: {context}

            Question: {question}
            '''


prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

consulta = 'What are the endnotes of the briefing?'

in_chain = {'context': recuperador, 'question': RunnablePassthrough()} | prompt | modelo | parser


respuesta = in_chain.invoke(consulta)
respuesta.split("\n")

from langchain_openai import OpenAI

input_model = OpenAI(temperature=0, max_tokens=1024)

#empezamos a preparar la creacion de pptx

template = '''
            We have provided  information below.
            Given this information, please generate a python code with python-pptx for three 
            slide presentation with this information. 
            
            Put the title in the first slide, 
            5 bullet points in the second slide and another 5 bullet in the third slide.
            Put list number in each bullet point.
                        
            Separate the bullet points into separate texts with line separator.
            Set font size to 20 for each bullet point. 
            Save the file in ../pptx/Red Sea Security Threats.pptx path

            Information: {context}
            '''


prompt = ChatPromptTemplate.from_template(template)

out_chain = prompt | input_model | parser

output = out_chain.invoke({'context': respuesta})

output.split('\n')

# ejecutando el codigo de python

exec(output)