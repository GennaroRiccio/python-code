import os
from langchain_community.utilities import SQLDatabase
from langchain.sql_database import SQLDatabaseChain
from langchain.llms import OpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

class SQLServerLangChain:
    def __init__(self, server, database, username=None, password=None, use_windows_auth=True):
        """
        Inizializza la connessione a SQL Server
        
        Args:
            server: Nome del server SQL Server
            database: Nome del database
            username: Username (se non si usa autenticazione Windows)
            password: Password (se non si usa autenticazione Windows)
            use_windows_auth: True per usare autenticazione Windows
        """
        if use_windows_auth:
            self.connection_string = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        else:
            self.connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        # Creare connessione database
        self.db = SQLDatabase.from_uri(self.connection_string)
        
    def get_tables(self):
        """Restituisce lista delle tabelle disponibili"""
        return self.db.get_table_names()
    
    def get_table_schema(self, table_name):
        """Restituisce schema di una tabella specifica"""
        return self.db.get_table_info(table_names=[table_name])
    
    def execute_query(self, query):
        """Esegue una query SQL diretta"""
        try:
            result = self.db.run(query)
            return result
        except Exception as e:
            return f"Errore nell'esecuzione della query: {str(e)}"
    
    def setup_agent(self, openai_api_key):
        """Configura l'agente SQL per query in linguaggio naturale"""
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        
        self.agent_executor = create_sql_agent(
            llm=llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def natural_language_query(self, question):
        """Esegue query usando linguaggio naturale"""
        if not hasattr(self, 'agent_executor'):
            return "Errore: Agente non configurato. Usa setup_agent() prima."
        
        try:
            response = self.agent_executor.run(question)
            return response
        except Exception as e:
            return f"Errore nella query: {str(e)}"
    
    def query_to_dataframe(self, query):
        """Esegue query e restituisce risultati come DataFrame pandas"""
        try:
            # Usa la connessione SQLAlchemy sottostante
            engine = self.db._engine
            df = pd.read_sql_query(query, engine)
            return df
        except Exception as e:
            print(f"Errore nel creare DataFrame: {str(e)}")
            return None

# Esempio di utilizzo
if __name__ == "__main__":
    # Configurazione
    sql_reader = SQLServerLangChain(
        server="localhost",  # o il tuo server SQL Server
        database="AdventureWorks2019",  # nome del tuo database
        use_windows_auth=True  # usa autenticazione Windows
    )
    
    # Ottenere lista tabelle
    print("Tabelle disponibili:")
    tables = sql_reader.get_tables()
    for table in tables[:5]:  # primi 5 tabelle
        print(f"- {table}")
    
    # Ottenere schema di una tabella
    if tables:
        schema = sql_reader.get_table_schema(tables[0])
        print(f"\nSchema della tabella {tables[0]}:")
        print(schema)
    
    # Eseguire query SQL diretta
    query = "SELECT TOP 5 * FROM Person.Person"
    print(f"\nRisultato query: {query}")
    result = sql_reader.execute_query(query)
    print(result)
    
    # Ottenere risultati come DataFrame
    df = sql_reader.query_to_dataframe("SELECT COUNT(*) as total_records FROM Person.Person")
    if df is not None:
        print(f"\nDataFrame risultato:")
        print(df)
    
    # Per usare l'agente con linguaggio naturale (richiede OpenAI API key)
    # sql_reader.setup_agent("your-openai-api-key")
    # response = sql_reader.natural_language_query("Quanti record ci sono nella tabella Person?")
    # print(response)