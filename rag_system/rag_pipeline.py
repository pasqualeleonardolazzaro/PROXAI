from rag_system.database import neo4j_client
from rag_system.indexing import faiss_indexer
from rag_system.retrieval import semantic_retriever, structural_expander, score_combiner, graph_expander
from rag_system.preprocessing import context_formatter
from rag_system.llm import generator
from rag_system.config import settings
from langchain_core.messages import HumanMessage, AIMessage


class RAGPipeline:
    """
    Optimized RAG pipeline that loads resources once and reuses them.
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
        
        print("RAG Pipeline ready!")
    
    def query(self, query, session_id="default_session"):
        """
        Execute a query through the RAG pipeline.
        """
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
        full_context = context_formatter.format_context(expanded_contexts)

        # Get chat history for this session
        history_messages = self.generator2.get_history(session_id)

        # Generate answer
        answer_object = self.generator2.runnable.invoke({
            "input": query,
            "context": full_context,
            "history": history_messages
        })
        
        # Update history
        self.generator2.add_to_history(session_id, query, answer_object.content)
        
        return answer_object.content, full_context
    
    def close(self):
        """Clean up resources."""
        if self.neo4j_driver:
            self.neo4j_driver.close()


# Global instance (lazy initialization)
_pipeline_instance = None


def get_pipeline():
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


def query_pipeline(query, session_id="default_session"):
    """
    Convenience function that maintains backwards compatibility.
    """
    pipeline = get_pipeline()
    return pipeline.query(query, session_id)