from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from rag_system.config import settings

def pivot_search(query, semantic_index, meta, top_k=settings.TOP_K):
    """
    Performs a semantic search to find the initial pivot nodes.
    """
    st = SentenceTransformer(settings.TEXT_EMBED_MODEL)
    query_sem_vec = st.encode(query)
    q_vec = normalize(query_sem_vec.reshape(1, -1), norm='l2').astype("float32")

    D_sem, I_sem = semantic_index.search(q_vec, top_k)
    
    pivot_hits = []
    for score, idx in zip(D_sem[0], I_sem[0]):
        if idx >= 0:
            pivot_hits.append({'id': meta["ids"][idx], 'score': score, 'idx': idx})
            
    return pivot_hits