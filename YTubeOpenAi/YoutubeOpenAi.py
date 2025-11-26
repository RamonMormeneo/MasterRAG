# importamos la API KEY

import os                           # libreria del sistema operativo
from dotenv import load_dotenv      # carga variables de entorno 


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langchain_openai.chat_models import ChatOpenAI   # conexión de LangChain a los modelos de OpenAI

modelo = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo')

# invocamos al modelo para que genere la respuesta

respuesta = modelo.invoke('¿Quién ganó la liga española en 2020?')

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# creamos la cadena

cadena = modelo | parser

# invocamos a la cadena dándole la consulta para que realice todo el flujo

cadena.invoke('¿Quién ganó la liga española en 2020?')

from langchain.prompts import ChatPromptTemplate

# gpt3.5 funciona mejor en inglés, hagamos un ejemplo en esa lengua


# plantilla de texto con un contexto y una pregunta
plantilla = '''
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            '''

# carga de la plantilla en el prompt
prompt = ChatPromptTemplate.from_template(plantilla)


# formato de salida del prompt
prompt.format(context='The building where Pepe lives is green.', 
              question="What color is Pepe's house?").split('\n')

# creamos la cadena añadiendo la plantilla del prompt en el primer paso

cadena = prompt | modelo | parser

# invocamos a la cadena

cadena.invoke({'context': 'The building where Pepe lives is green.',
               'question': "What color is Pepe's house?"})

# creamos la plantilla de traducción con la respuesta de la cadena y el lenguaje de salida

prompt_traductor = ChatPromptTemplate.from_template('Translate {answer} to {language}')

# creamos la nueva cadena basada en la anterior a la cual le damos el lenguaje al que queremos traducir

cadena_traducida = (
    {'answer': cadena, 'language': lambda x: x['language']} | prompt_traductor | modelo | parser 
)

# invocamos a la nueva cadena, dándole el contexto, la pregunta y el lenguaje de salida

cadena_traducida.invoke({'context': 'The building where Pepe lives is green.',
                         'question': "What color is Pepe's house?",
                         'language': 'Spanish',
                         })

from pytube import YouTube

import whisper

# definimos la url del video: Planned Chaos - by Ludwig von Mises - (Full Audiobook) (3:09:32)

VIDEO_URL = 'https://www.youtube.com/watch?v=7EnHeZXLzTc'

# usamos pytube para extraer el video

youtube = YouTube(VIDEO_URL)

# ahora extraemos el audio desde el video

audio = youtube.streams.filter(only_audio=True).first()

audio

# se carga el modelo base de Whisper en local, 139M

modelo_whisper = whisper.load_model('base')

# descripción del modelo whisper

modelo_whisper

import tempfile    # para manejo de archivos temporales


# ruta de guardado del archivo de texto

RUTA_TXT = 'txt/transcripcion.txt'

# si el archivo de texto no existe...
if not os.path.exists(RUTA_TXT):
    
    # abrimos el directorio temporal...
    with tempfile.TemporaryDirectory() as dir_temporal:
        
        # descargamos el audio de YouTube...
        archivo_audio = audio.download(output_path=dir_temporal)
        
        # y Whisper transcribe el audio a texto, en 32 bits (fp16=False)
        transcripcion = modelo_whisper.transcribe(archivo_audio, fp16=False)['text'].strip()
        
        
        # se guarda el archivo de texto
        with open(RUTA_TXT, 'w') as archivo_texto:
            archivo_texto.write(transcripcion)

# cargamos el archivo de texto para usarlo en la cadena

with open(RUTA_TXT, 'r') as archivo_texto:
    
    transcripcion = archivo_texto.read()

# 1000 primeros caracteres del texto
    
transcripcion[:1000]

try:
    cadena.invoke({
        'context': transcripcion,
        'question': 'What are the key points of this book? Put them in a bullet point list.'})
    
except Exception as e:
    print(e)

#esto da error porque transcipcion es demasiado grande


from langchain_community.document_loaders import TextLoader

loader = TextLoader(RUTA_TXT)

documento_texto = loader.load()


len(documento_texto)

type(documento_texto[0])

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)

documentos = splitter.split_documents(documento_texto)

# número de trozos de texto desde la transcripción completa

len(documentos)   

from langchain_openai.embeddings import OpenAIEmbeddings


vectorizador = OpenAIEmbeddings()

# vectorizamos una frase

consulta = vectorizador.embed_query('Hola que tal, esto es una clase de IA.')


# longitud del vector

print(f'Longitud del vector consulta: {len(consulta)}\n')


# primero 5 elementos del vector

print(consulta[:5])

# ahora vectoricemos dos frase más 

frase1 = vectorizador.embed_query('¿Hoy estuviste estudiando o no?')

frase2 = vectorizador.embed_query('Estamos en clase, hay que ponerse a estudiar.')

from sklearn.metrics.pairwise import cosine_similarity

similitud_frase1 = cosine_similarity([consulta], [frase1])[0][0]

similitud_frase2 = cosine_similarity([consulta], [frase2])[0][0]

similitud_frase1, similitud_frase2


from langchain_community.vectorstores import DocArrayInMemorySearch


# a docarray le pasamos los texto y el modelo de embedding

local_db = DocArrayInMemorySearch.from_documents(documentos, vectorizador)

# podemos extraer los k vectores más relevates con su similitud, donde k es el nº de trozos que queremos 

consulta = 'What are the key points of this book? Put them in a bullet point list.'

local_db.similarity_search_with_score(query=consulta, k=1)

# también podemos extraer los k vectores más relevates sin similitud

local_db.similarity_search(query=consulta, k=1)

# o definir un recuperador de la base de datos para invocarlo que por defecto devuelve los 4 más relevantes

recuperador = local_db.as_retriever()

recuperador.invoke(consulta)

len(recuperador.invoke(consulta))

# aunque podemos definirlo para extraer los que queramos

# o definir un recuperador de la base de datos para invocarlo que por defecto devuelve los 4 más parecidos

recuperador = local_db.as_retriever(search_kwargs={'k': 5})

len(recuperador.invoke(consulta))

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ejecutamos el RunnableParallel para obtener el contexto

contexto_recuperado = RunnableParallel(context=recuperador, question=RunnablePassthrough())

contexto_recuperado.invoke(consulta)

# creamos la cadena con el contexto recuperado desde la bade de datos

cadena = contexto_recuperado | prompt | modelo | parser

cadena.invoke(consulta)

#con chroma

from langchain_community.vectorstores import Chroma

# guardado en disco

chroma_db = Chroma.from_documents(documentos, vectorizador, persist_directory='save_db')

docs = chroma_db.similarity_search(consulta)

docs[0]

# carga desde disco

chroma_db = Chroma(persist_directory='save_db', embedding_function=vectorizador)

docs = chroma_db.similarity_search(consulta)

docs[0]

# cadena con traducción y con la recuperacion de ChromaDB con 5 documentos

cadena =  prompt | modelo | parser

cadena_traducida = (
    {'answer': cadena, 'language': lambda x: x['language']} | prompt_traductor | modelo | parser
)

cadena_traducida.invoke({'context': chroma_db.similarity_search(consulta, k=5), 
                         'question': consulta,
                         'language': 'Spanish'}).split('\n')


#resumen

# librerias
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from pytube import YouTube
import whisper
import tempfile

import os                           
from dotenv import load_dotenv      

load_dotenv()


# importamos la api key de OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# iniciamos modelo gpt3.5 turbo
modelo = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo')



# plantilla de texto con un contexto y una pregunta
plantilla = '''
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            '''

# carga de la plantilla en el prompt
prompt = ChatPromptTemplate.from_template(plantilla)


# traduccion
prompt_traductor = ChatPromptTemplate.from_template('Translate {answer} to {language}')


# extraccion de datos
# definimos la url del video
VIDEO_URL = 'https://www.youtube.com/watch?v=7EnHeZXLzTc'

# usamos pytube para extraer el video
youtube = YouTube(VIDEO_URL)
audio = youtube.streams.filter(only_audio=True).first()

# se carga el modelo base de Whisper 
modelo_whisper = whisper.load_model('base')


# ruta de guardado del archivo de texto
RUTA_TXT = 'txt/transcripcion.txt'

# si el archivo de texto no existe...
if not os.path.exists(RUTA_TXT):
    
    # abrimos el directorio temporal...
    with tempfile.TemporaryDirectory() as dir_temporal:
        
        # descargamos el audio de YouTube...
        archivo_audio = audio.download(output_path=dir_temporal)
        
        # y Whiper transcribe el audio a texto.
        transcripcion = modelo_whisper.transcribe(archivo_audio, fp16=False)['text'].strip()
        
        
        # se guarda el archivo de texto
        with open(RUTA_TXT, 'w') as archivo_texto:
            archivo_texto.write(transcripcion)


# cargamos el archivo de texto para usarlo en la cadena
with open(RUTA_TXT, 'r') as archivo_texto:
    transcripcion = archivo_texto.read()


# transformacion del dato
loader = TextLoader(RUTA_TXT)
documento_texto = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
documentos = splitter.split_documents(documento_texto)


vectorizador = OpenAIEmbeddings(api_key=OPENAI_API_KEY)



# guardado en disco, no sería necesario, hacer aparte
chroma_db = Chroma.from_documents(documentos, vectorizador, persist_directory='save_db')


# carga desde disco
chroma_db = Chroma(persist_directory='save_db', embedding_function=vectorizador)




# consulta
consulta = 'What are the key points of this book? Put them in a bullet point list.'


# parser a string
parser = StrOutputParser()




# cadena con traducción y con la recuperacion de ChromaDB con 5 documentos
cadena =  prompt | modelo | parser


cadena_traducida = (
    {'answer': cadena, 'language': lambda x: x['language']} | prompt_traductor | modelo | parser
)

cadena_traducida.invoke({'context': chroma_db.similarity_search(consulta, k=5), 
                         'question': consulta,
                         'language': 'Spanish'}).split('\n')