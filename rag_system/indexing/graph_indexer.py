from rag_system.database import neo4j_client
from rag_system.embeddings import structural_embedder, semantic_embedder
from rag_system.indexing import faiss_indexer

def build_indexes():
    """
    Main function to orchestrate the full graph indexing pipeline.
    """
    driver = neo4j_client.connect_neo4j()
    
    print("Fetching graph data from Neo4j...")
    nodes, edges = neo4j_client.fetch_graph(driver)
    print(f"Fetched {len(nodes)} nodes and {len(edges)} edges.")

    print("Computing structural (Node2Vec) embeddings...")
    node2v_emb = structural_embedder.compute_node2vec_embeddings(driver)
    
    print("Computing semantic (SentenceTransformer) embeddings...")
    text_emb = semantic_embedder.compute_text_embeddings(nodes)
    
    print("Building and storing FAISS vector indexes...")
    faiss_indexer.build_and_store_vector_indexes(nodes, node2v_emb, text_emb)
    
    driver.close()
    print("\nIndexing complete!")