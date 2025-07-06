from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import json

class OllamaSQLAgent:
    def __init__(self, server, database, username=None, password=None, use_windows_auth=True):
        """
        Inizializza l'agente SQL con Ollama
        """
        if use_windows_auth:
            self.connection_string = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        else:
            self.connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        self.db = SQLDatabase.from_uri(self.connection_string)
        self.llm = None
        self.agent_executor = None
    
    def setup_ollama(self, model_name="llama3", base_url="http://localhost:11434"):
        """Configura Ollama con parametri ottimizzati per SQL"""
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_ctx=4096,
            num_predict=1024,
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.3,
            system="""You are an expert SQL Server assistant. 
            Rules:
            - Use TOP instead of LIMIT for SQL Server
            - Use square brackets for names with spaces
            - Be precise and explain your reasoning
            - Return clear, executable SQL queries"""
        )
        
        # Creare l'agente SQL
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10,
            max_execution_time=120
        )
    
    def ask_question(self, question, context=None):
        """Fai una domanda in linguaggio naturale"""
        if not self.agent_executor:
            return "Errore: Configurare prima Ollama con setup_ollama()"
        
        # Aggiungi contesto se fornito
        if context:
            enhanced_question = f"""
            Contesto: {context}
            
            Domanda: {question}
            
            Genera una query SQL Server appropriata e eseguila.
            """
        else:
            enhanced_question = question
        
        try:
            response = self.agent_executor.run(enhanced_question)
            return response
        except Exception as e:
            return f"Errore: {str(e)}"
    
    def generate_sql_only(self, question):
        """Genera solo la query SQL senza eseguirla"""
        if not self.llm:
            return "Errore: Configurare prima Ollama con setup_ollama()"
        
        # Template per generare solo SQL
        sql_template = PromptTemplate(
            input_variables=["question", "schema"],
            template="""
            Basandoti su questo schema del database:
            {schema}
            
            Genera SOLO la query SQL Server per rispondere a questa domanda:
            {question}
            
            Regole:
            - Usa TOP invece di LIMIT
            - Usa square brackets per nomi con spazi
            - Restituisci SOLO il codice SQL, niente spiegazioni
            """
        )
        
        # Ottieni schema delle tabelle
        schema = self.db.get_table_info()
        
        # Crea la chain
        sql_chain = LLMChain(llm=self.llm, prompt=sql_template)
        
        try:
            result = sql_chain.run(question=question, schema=schema)
            return result.strip()
        except Exception as e:
            return f"Errore nella generazione SQL: {str(e)}"
    
    def explain_query(self, sql_query):
        """Spiega cosa fa una query SQL"""
        if not self.llm:
            return "Errore: Configurare prima Ollama con setup_ollama()"
        
        explain_template = PromptTemplate(
            input_variables=["query"],
            template="""
            Spiega in italiano cosa fa questa query SQL Server:
            
            {query}
            
            Fornisci una spiegazione chiara e concisa.
            """
        )
        
        explain_chain = LLMChain(llm=self.llm, prompt=explain_template)
        
        try:
            explanation = explain_chain.run(query=sql_query)
            return explanation
        except Exception as e:
            return f"Errore nella spiegazione: {str(e)}"
    
    def get_table_summary(self, table_name):
        """Ottieni un riassunto dei dati in una tabella"""
        try:
            # Query per statistiche base
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT *) as unique_rows
            FROM [{table_name}]
            """
            
            # Schema della tabella
            schema = self.db.get_table_info(table_names=[table_name])
            
            # Esegui query statistiche
            stats = self.db.run(stats_query)
            
            # Genera riassunto con LLM
            if self.llm:
                summary_template = PromptTemplate(
                    input_variables=["table_name", "schema", "stats"],
                    template="""
                    Tabella: {table_name}
                    
                    Schema:
                    {schema}
                    
                    Statistiche:
                    {stats}
                    
                    Fornisci un riassunto breve e utile di questa tabella.
                    """
                )
                
                summary_chain = LLMChain(llm=self.llm, prompt=summary_template)
                summary = summary_chain.run(
                    table_name=table_name,
                    schema=schema,
                    stats=stats
                )
                
                return {
                    "schema": schema,
                    "statistics": stats,
                    "summary": summary
                }
            else:
                return {
                    "schema": schema,
                    "statistics": stats,
                    "summary": "LLM non configurato per il riassunto"
                }
                
        except Exception as e:
            return f"Errore nel riassunto tabella: {str(e)}"

# Esempio di utilizzo
if __name__ == "__main__":
    # Inizializza l'agente
    agent = OllamaSQLAgent(
        server="localhost",
        database="AdventureWorks2019",
        use_windows_auth=True
    )
    
    # Configura Ollama
    print("Configurazione Ollama...")
    agent.setup_ollama(model_name="llama3")
    
    # Test di base
    print("\n=== Test domanda semplice ===")
    response = agent.ask_question("Quante persone ci sono nel database?")
    print(response)
    
    # Genera solo SQL
    print("\n=== Generazione SQL ===")
    sql = agent.generate_sql_only("Mostra i primi 5 clienti ordinati per nome")
    print(f"SQL generato: {sql}")
    
    # Spiega una query
    print("\n=== Spiegazione query ===")
    explanation = agent.explain_query("SELECT TOP 10 * FROM Person.Person WHERE LastName LIKE 'A%'")
    print(explanation)
    
    # Riassunto tabella
    print("\n=== Riassunto tabella ===")
    summary = agent.get_table_summary("Person.Person")
    print(json.dumps(summary, indent=2, ensure_ascii=False))