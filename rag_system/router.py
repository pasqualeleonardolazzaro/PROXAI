from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryRouter:
    def __init__(self, llm):
        self.llm = llm
        
        system_prompt = """
        You are a query classifier for a Graph RAG system.
        Classify the user's input into one of two categories:
        
        1. "ANALYTIC": The user is asking for an aggregation (count, sum, average),  a complete list of items (e.g., "List all...", "What are the...", "How many..."). 
           - These queries need a database query (Cypher).
           
        2. "SEARCH": The user is asking for information about specific entities, descriptions, relationships, or concepts (e.g., "Who is...", "Tell me about...", "What is the connection between...").
           - These queries need Vector Search and Context Expansion.

        Output ONLY the word "ANALYTIC" or "SEARCH".
        """
        
        self.chain = (
            ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            | llm
            | StrOutputParser()
        )

    def route(self, query):
        decision = self.chain.invoke({"input": query}).strip().upper()
        # Default to SEARCH if the LLM output is weird
        if "ANALYTIC" in decision:
            return "ANALYTIC"
        return "SEARCH"