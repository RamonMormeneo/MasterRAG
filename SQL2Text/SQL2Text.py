import warnings
warnings.filterwarnings('ignore')
# carga de la api key desde dotenv

import os                           
from dotenv import load_dotenv     


load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # 'sk-.....'

from sqlalchemy import create_engine, text

# string de conexion

URI = 'mysql+pymysql://root:password@localhost:3306/sakila'

# conexion a SQL 

cursor = create_engine(URI).connect()

# tablas de la base de datos

tablas = cursor.execute(text('show tables;')).all()

tablas = [e[0] for e in tablas]

tablas

from langchain import SQLDatabase

# conexion a langchain con todas las tablas y la primera fila de cada una

db = SQLDatabase.from_uri(URI,
                          sample_rows_in_table_info=1, 
                          include_tables=tablas)

# estructura completa que se usara como contexto

print(db.table_info)

# importamos el modelo

from langchain_openai import OpenAI

input_model = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

# prueba de uso

input_model.invoke('hola')

from langchain.chains import create_sql_query_chain

# cadena por defecto

database_chain = create_sql_query_chain(input_model, db)

database_chain

# prompt custom

from langchain_core.prompts import PromptTemplate

template = '''You are a MySQL expert. Given an input question, 
              first create a syntactically correct MySQL query to run, 
              then look at the results of the query and return the answer to the input question.
              Unless the user specifies in the question a specific number of examples to obtain, 
              query for at most {top_k} results using the LIMIT clause as per MySQL. 
              You can order the results to return the most informative data in the database.
              Never query for all columns from a table. You must query only the columns that 
              are needed to answer the question. 
              Wrap each column name in backticks (`) to denote them as delimited identifiers.
              Pay attention to use only the column names you can see in the tables below. 
              Be careful to not query for columns that do not exist. 
              Also, pay attention to which column is in which table.
              Pay attention to use CURDATE() function to get the current date, 
              if the question involves "today".
              
              Use the following format:
              
              Question: Question here
              
              SQLQuery: SQL Query to run
              
              SQLResult: Result of the SQLQuery
              
              Answer: Final answer here
              
              Only use the following tables:
              
              {table_info}
              
              Question: {input}'''




custom_prompt = PromptTemplate(input_variables=['input', 'table_info', 'top_k', 'dialect'],
                               template=template)

database_chain = create_sql_query_chain(input_model, db, prompt=custom_prompt)

# nuestra pregunta

prompt = '¿Qué actores tienen de primer nombre SCARLETT?'

# traduccion a ingles, funciona mejor

prompt = input_model.invoke(f'traduce al ingles: {prompt}')

prompt

# llamada al modelo creador de queries

sql_query = database_chain.invoke({'question': prompt})

# query SQL creada

sql_query

# respuesta de la query que usaremos como contexto

contexto = cursor.execute(text(sql_query)).all()

contexto

from langchain_openai.chat_models import ChatOpenAI   

output_model = ChatOpenAI(model='gpt-4-turbo')

# le damos la pregunta original y el contexto extraido desde SQL

final_prompt = f'''Given the next context, answer the cuestion: 
                    
                    context: {contexto}, 
                    
                    question: {prompt}
                    
                    Give the answer in Spanish.
                    
                    '''

respuesta_final = output_model.invoke(final_prompt).content

respuesta_final

#Codigo completo

# librerias   


import warnings
warnings.filterwarnings('ignore')                     # para quitar avisos

from sqlalchemy import create_engine, text            # conexion SQL y text para queries

from langchain import SQLDatabase                     # conexion SQL a LangChain
from langchain_core.prompts import PromptTemplate     # creacion de prompts
from langchain.chains import create_sql_query_chain   # cadana de creacion de queries SQL
from langchain_openai import OpenAI                   # modelo OpenAI
from langchain_openai.chat_models import ChatOpenAI   # modelo chat OpenAI


import os                                             # libreria del sistema
from dotenv import load_dotenv                        # carga de variables de entorno
 

# variables de entorno
load_dotenv()


# api key OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# string de conexion servidor SQL
URI = 'mysql+pymysql://root:password@localhost:3306/sakila'



# prompt inicial
input_model = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

prompt = '¿Qué actores tienen como apellido JOHANSSON?'

prompt = input_model.invoke(f'Traduce al ingles: {prompt}')

prompt = prompt 


# conexion a base de datos
cursor = create_engine(URI).connect()

tablas = cursor.execute(text('show tables;')).all()
tablas = [e[0] for e in tablas]

db = SQLDatabase.from_uri(URI, sample_rows_in_table_info=1, include_tables=tablas)



# definion del prompt para generar query SQL
sql_prompt = '''You are a MySQL expert. Given an input question, 
          first create a syntactically correct MySQL query to run, 
          then look at the results of the query and return the answer to the input question.
          Unless the user specifies in the question a specific number of examples to obtain, 
          query for at most {top_k}0 results using the LIMIT clause as per MySQL. 
          You can order the results to return the most informative data in the database.
          Never query for all columns from a table. You must query only the columns that 
          are needed to answer the question. 
          Wrap each column name in backticks (`) to denote them as delimited identifiers.
          Pay attention to use only the column names you can see in the tables below. 
          Be careful to not query for columns that do not exist. 
          Also, pay attention to which column is in which table.
          Pay attention to use CURDATE() function to get the current date, 
          if the question involves "today".

          Use the following format:

          Question: Question here

          SQLQuery: SQL Query to run

          SQLResult: Result of the SQLQuery

          Answer: Final answer here

          Only use the following tables:

          {table_info}

          Question: {input}'''


sql_prompt = PromptTemplate(input_variables=['input', 'table_info', 'top_k', 'dialect'],
                            template=sql_prompt)


# creacion de query SQL
database_chain = create_sql_query_chain(input_model, db, prompt=sql_prompt)

sql_query = database_chain.invoke({'question': prompt})


# ejecucion de la query SQL
contexto = cursor.execute(text(sql_query)).all()


# respuesta final 
output_model = ChatOpenAI(model='gpt-4-turbo')

final_prompt = f'''Given the next query and context, answer the cuestion:

               query: {sql_query},

               context: {contexto}, 

               question: {prompt}.

               If the context is a number, answer the question with it.

               Give the answer in Spanish.
               '''

respuesta = output_model.invoke(final_prompt).content


print(respuesta)