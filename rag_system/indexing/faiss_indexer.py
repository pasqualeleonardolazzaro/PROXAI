import os
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from rag_system.config import settings


def build_and_store_vector_indexes(nodes, node2vec_emb, text_emb):
    """
    Builds two FAISS indexes:
    - Semantic index: one entry per (node, text_chunk)
    - Structural index: one entry per node (unchanged)

    Metadata stores full traceability back to node + chunk.
    """
    semantic_vectors = []
    structural_vectors = []
    metadatas = []
    ids = []  # FAISS ids (one per semantic chunk)

    valid_node_ids = sorted(list(node2vec_emb.keys()))

    # --- STRUCTURAL PART  ---
    
    struct_xb_list = []
    struct_ids = []

    for nid in valid_node_ids:
        
        s_vec_norm = normalize(node2vec_emb[nid].reshape(1, -1), norm='l2')[0]
        struct_xb_list.append(s_vec_norm.astype("float32"))
        struct_ids.append(str(nid))


    # --- SEMANTIC PART  ---
    for nid in valid_node_ids:
        if nid not in text_emb:
            continue

        chunk_texts = text_emb[nid]["texts"]
        chunk_vecs = text_emb[nid]["vecs"]

        for chunk_id, (txt, vec) in enumerate(zip(chunk_texts, chunk_vecs)):
            vec_norm = normalize(vec.reshape(1, -1), norm='l2')[0]

            # One FAISS entry per (node, chunk)
            ids.append(str(nid)) 
            semantic_vectors.append(vec_norm.astype("float32"))

            # Store metadata
            metadatas.append({
                "node_id": str(nid),
                "chunk_id": chunk_id,
                "chunk_text": txt,
                "labels": nodes[nid]["labels"],
                "props": nodes[nid]["props"]
            })


    # --- SAVE INDEXES ---

    os.makedirs(settings.INDEX_PATH, exist_ok=True)

    # SEMANTIC
    sem_xb = np.vstack(semantic_vectors)
    semantic_index = faiss.IndexFlatIP(sem_xb.shape[1])
    semantic_index.add(sem_xb)
    faiss.write_index(semantic_index, settings.SEMANTIC_INDEX_FILE)
    print(f"Saved semantic index to {settings.SEMANTIC_INDEX_FILE}")

    # STRUCTURAL (unchanged)
    struct_xb = np.vstack(struct_xb_list)
    structural_index = faiss.IndexFlatIP(struct_xb.shape[1])
    structural_index.add(struct_xb)
    faiss.write_index(structural_index, settings.STRUCTURAL_INDEX_FILE)
    print(f"Saved structural index to {settings.STRUCTURAL_INDEX_FILE}")

    # --- METADATA ---
    with open(settings.META_FILE, "wb") as f:
        pickle.dump({
            "ids": ids,                     # Semantic IDs (with duplicates for chunks)
            "struct_ids": struct_ids,       # Structural IDs (unique, 1-to-1 with structural_vectors)
            "metadatas": metadatas,
            "semantic_vectors": sem_xb,
            "structural_vectors": struct_xb,
        }, f)

    print(f"Saved metadata and vectors to {settings.META_FILE}")


def load_indexes_and_meta():
    """Loads FAISS indexes and the metadata file."""
    semantic_index = faiss.read_index(settings.SEMANTIC_INDEX_FILE)
    structural_index = faiss.read_index(settings.STRUCTURAL_INDEX_FILE)
    with open(settings.META_FILE, "rb") as f:
        meta = pickle.load(f)
    return semantic_index, structural_index, meta