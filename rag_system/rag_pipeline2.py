from rag_system.database import neo4j_client
from rag_system.indexing import faiss_indexer
from rag_system.retrieval import semantic_retriever, structural_expander, score_combiner, graph_expander,cypher_retriever
from rag_system.preprocessing import context_formatter
from rag_system.llm import generator
from rag_system.config import settings
from langchain_core.messages import HumanMessage, AIMessage
from rag_system.router import QueryRouter 

class RAGPipeline:
    
    """
     RAG pipeline
    """
    def __init__(self):
        """Initialize and cache all resources."""
        print("Initializing RAG Pipeline...")
        
        # Load indexes once
        self.semantic_index, self.structural_index, self.meta = faiss_indexer.load_indexes_and_meta()
        
        # Keep Neo4j driver open (connection pooling handles efficiency)
        self.neo4j_driver = neo4j_client.connect_neo4j()
        
        # Generator2 is already initialized in generator2.py
        self.generator2 = generator

        #initialize cypher_retriever
        self.cypher_retriever=cypher_retriever.CypherRetriever(self.neo4j_driver,self.generator2.llm)

        # initialize queryRouter (decides what to use: text-to-cypher or embeddings)
        self.router = QueryRouter(self.generator2.llm)
        
        print("RAG Pipeline ready!")

    # Helper to decide if result is actually useful
    def _is_meaningful_result(self,result):
        # If we have nodes not empty it's relevant
        if result['nodes']:
            return True
            
        # If we have facts, check if they are just "0" or "None"
        if result['facts']:
            for fact in result['facts']:
                val_part = fact.split(":")[-1].strip()
                
                # If non-zero values found Return True
                if val_part not in ["0", "0.0", "None", "[]", "null"]:
                    return True
                    
        return False
        

    def query(self, query, session_id="default_session"):
        # Decide Strategy
        route = self.router.route(query)
        print(f"--- Query Routed to: {route} ---")

        context_str = ""
        

        # PATH 1: ANALYTIC does text-to-cypher

        if route == "ANALYTIC":
            # Run strictly Cypher
            result = self.cypher_retriever.retrieve(query)
            # Check if we actually got results
            if self._is_meaningful_result(result):
                print("--- Analytic Answer Found ---")
                context_str = context_formatter.format_analytic_result(result)
            else:
                print("--- Analytic Path Failed (0 results), falling back to Search ---")
                route = "SEARCH" # Fallback 


        # PATH 2: SEARCH uses embeddings

        if route == "SEARCH":

                # Pivot Search (Semantic)
            pivot_hits = semantic_retriever.pivot_search(
                query, self.semantic_index, self.meta
            )
            
            # Candidate Expansion (Structural)
            candidate_nodes = structural_expander.expand_candidates(
                pivot_hits, self.structural_index, self.meta
            )
            
            # Re-ranking and Consolidation
            final_hits = score_combiner.rerank_and_consolidate(candidate_nodes, self.meta)

            # Context Generation with graph expansion
            expanded_contexts = []
            for hit in final_hits:
                graph_data = graph_expander.cypher_expand_context(
                    self.neo4j_driver, hit["id"], hops=settings.EXPAND_HOPS
                )
                expanded_contexts.append({
                    "node_id": hit["id"],
                    "score": hit["score"],
                    "meta": hit["meta"],
                    "graph_data": graph_data
                })

            # Format context for LLM
            context_str = context_formatter.format_context(expanded_contexts)
            

        # generate answer

        history_messages = self.generator2.get_history(session_id)
        answer = self.generator2.runnable.invoke({
            "input": query,
            "context": context_str,
            "history": history_messages
        })
        
        self.generator2.add_to_history(session_id, query, answer.content)
        return answer.content, context_str

    
# Global instance (lazy initialization)
_pipeline_instance = None


def get_pipeline():
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance