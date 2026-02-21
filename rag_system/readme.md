# RAG System (Neo4j + FAISS + LLM Routing)

This RAG system blends semantic search, graph structural embeddings, and LLM-powered routing to retrieve and synthesize answers from a Neo4j knowledge graph.

## Key Features
- Dual FAISS indexes:
  - **Semantic Index**: MiniLM embeddings of textual properties
  - **Structural Index**: Node2Vec embeddings from Neo4j GDS
- Two retrieval routes:
  - **Analytic Route**: LLM-generated Cypher queries for list/aggregation/date queries
  - **Search Route**: Vector search + structural expansion + 2-hop graph context expansion
- LLM-powered:
  - Query classification
  - Cypher generation
  - Final answer synthesis

## Retrieval Flow
1. LLM classifies the question → Analytic or Search route.
2. **Analytic Route**:
   - LLM generates Cypher from schema + question
   - Execute Cypher
   - If results: format & answer
   - Else: fallback to Search route
3. **Search Route**:
   - Embed query → semantic vector search
   - Structural re-ranking using node2vec embeddings
   - Expand 2-hop neighbors
   - Format graph context → LLM generates answer

## Technologies
- Neo4j  
- Neo4j GDS Node2Vec  
- FAISS  
- Sentence Transformers (MiniLM L6 v2)  
- Any LLM (router + generator)
