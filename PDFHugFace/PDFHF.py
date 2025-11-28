import os   # libreria del sistema operativo

# lista de archivos en la carpeta pdfs

os.listdir('pdfs') 

from langchain_community.document_loaders import PyPDFLoader

# carga de archivo PDF página a página

loader = PyPDFLoader('pdfs/memoria_consolidada_2022.pdf')

paginas = loader.load()

len(paginas)
from langchain_community.document_loaders import PyPDFDirectoryLoader

# carga de todos los archivos PDF página a página

loader = PyPDFDirectoryLoader('pdfs/')

paginas = loader.load()


# carga de todos los archivos PDF y realiza los chunks

loader = PyPDFDirectoryLoader('pdfs/')

documentos = loader.load_and_split()

len(documentos)

from langchain_huggingface import HuggingFaceEmbeddings

# inicializamos el modelo de embedding con Roberta

vectorizador = HuggingFaceEmbeddings(model_name='sentence-transformers/all-roberta-large-v1')

# realizamos una prueba para comprobar que funciona correctamente

vector = vectorizador.embed_query(documentos[10].page_content)

vector[:3]

from langchain_community.vectorstores import Chroma

# guardado en disco

chroma_db = Chroma.from_documents(documentos,                    # documentos de texto
                                  vectorizador,                  # modelo de embedding
                                  persist_directory='save_db'    # ruta de guardado
                                 )

# carga desde disco

chroma_db = Chroma(persist_directory='save_db', embedding_function=vectorizador)

# búsqueda por similitud con el retorno de los 5 documentos más relevantes

documentos = chroma_db.similarity_search('Derivados de activos no corrientes 2022', k=5)

documentos[0]

recuperador = chroma_db.as_retriever(search_type='mmr', 
                                     search_kwargs={'k': 20, 'lambda_mult': 0.25})

#plantilla de LLM
from langchain_core.prompts import ChatPromptTemplate

# plantilla de texto con un contexto y una pregunta
plantilla = '''
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            
            Don´t response with the prompt. Translate the answer to Spanish.
            '''


# carga de la plantilla en el prompt
prompt = ChatPromptTemplate.from_template(plantilla)

prompt

from dotenv import load_dotenv      # carga variables de entorno 

load_dotenv()


# importamos el token

HUGGINGFACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace



llm = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-beta',
                     task='text-generation',
                     huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                     
                     model_kwargs={'max_new_tokens': 512,
                                   'top_k': 30,
                                   'temperature': 0.1,
                                   'repetition_penalty': 1.03})


modelo = ChatHuggingFace(llm=llm)

# realizamos una prueba del modelo para ver que funciona

modelo.invoke('Capital de España')

consulta = '¿Cuál es el Balance de Situación Financiera Consolidado al 31 de diciembre de 2022?'


# cadena con la plantilla de prompt y el modelo, con LangChain Expression Language (LCEL)
cadena = prompt | modelo


# invocamos a la cadena con el contexto traido de la base de datos y la consulta 
respuesta = cadena.invoke({'context': chroma_db.similarity_search(consulta), 
                           'question': consulta})


# transformamos la respuesta del modelo 
respuesta.content.split('<|assistant|>')[1].strip()

from langchain_core.runnables import RunnablePassthrough

consulta = '¿Cuál es el Balance de Situación Financiera Consolidado al 31 de diciembre de 2022?'


# recuperamos los 5 documentos más relevantes de la base de datos
recuperador = chroma_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.25})


# cadena con el recuperador, la plantilla de prompt y el modelo
cadena = {'context': recuperador, 'question': RunnablePassthrough()} | prompt | modelo


# respuesta de la cadena
respuesta = cadena.invoke(consulta)


# transformamos la respuesta del modelo
respuesta.content.split('<|assistant|>')[1].strip()

#Resumen

# librerias
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
load_dotenv()

# importamos el token de huggingface
HUGGINGFACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')



# carga de todos los archivos PDF y realiza los chunks
loader = PyPDFDirectoryLoader('pdfs/')
documentos = loader.load_and_split()


# modelo embedding
vectorizador = HuggingFaceEmbeddings(model_name='sentence-transformers/all-roberta-large-v1')


# guardado en disco, no sería necesario, debería de hacerse aparte
chroma_db = Chroma.from_documents(documentos,                    # documentos de texto
                                  vectorizador,                  # modelo de embedding
                                  persist_directory="save_db"    # ruta de guardado
                                 )


# carga desde disco
chroma_db = Chroma(persist_directory='save_db', embedding_function=vectorizador)



# plantilla de texto con un contexto y una pregunta
plantilla = '''
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            
            Don´t response with the prompt. Translate the answer to Spanish.
            '''


# carga de la plantilla en el prompt
prompt = ChatPromptTemplate.from_template(plantilla)



# modelo de huggingface
llm = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-beta',
                     task='text-generation',
                     huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                     
                     model_kwargs={'max_new_tokens': 512,
                                   'top_k': 30,
                                   'temperature': 0.1,
                                   'repetition_penalty': 1.03})


modelo = ChatHuggingFace(llm=llm)


# consulta que hacemos
consulta = '¿Cuál es el Balance de Situación Financiera Consolidado al 31 de diciembre de 2022?'


# recuperamos los 5 documentos más relevantes de la base de datos
recuperador = chroma_db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 0.25})


# cadena con el recuperador, la plantilla de prompt y el modelo
cadena = {'context': recuperador, 'question': RunnablePassthrough()} | prompt | modelo


# respuesta de la cadena
respuesta = cadena.invoke(consulta)


# transformamos la respuesta del modelo
respuesta.content.split('<|assistant|>')[1].strip()

