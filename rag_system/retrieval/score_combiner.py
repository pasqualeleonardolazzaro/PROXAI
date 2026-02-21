from rag_system.config import settings

def rerank_and_consolidate(candidate_nodes, meta, top_k=settings.TOP_K):
    """
    Combines scores and re-ranks candidates to produce the final list of hits.
    """
    scored_candidates = {}
    for node_id, scores in candidate_nodes.items():
        sem_score = scores['semantic_score']
        # Average the structural scores for stability
        struct_score = sum(scores['structural_scores']) / len(scores['structural_scores']) if scores['structural_scores'] else 0
        
        # Weighted final score
        final_score = (sem_score * settings.SEMANTIC_WEIGHT) + (struct_score * settings.STRUCTURAL_WEIGHT)
        scored_candidates[node_id] = final_score

    # Sort candidates by their combined score
    sorted_candidates = sorted(scored_candidates.items(), key=lambda item: item[1], reverse=True)
    
    # Get metadata for the top K hits
    id_to_idx = {node_id: i for i, node_id in enumerate(meta["ids"])}
    final_hits = []
    for node_id, final_score in sorted_candidates[:top_k]:
        idx = id_to_idx[node_id]
        final_hits.append({"id": node_id, "score": final_score, "meta": meta["metadatas"][idx]})
        
    return final_hits