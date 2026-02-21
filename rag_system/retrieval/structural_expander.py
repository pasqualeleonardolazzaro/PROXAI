from rag_system.config import settings

def expand_candidatesOld(pivot_hits, structural_index, meta, top_k=settings.TOP_K):
    """
    Expands from pivot nodes to find structurally similar candidates.
    """
    candidate_nodes = {}
    
    for hit in pivot_hits:
        # Add the pivot node itself
        candidate_nodes[hit['id']] = candidate_nodes.get(hit['id'], {'semantic_score': 0, 'structural_scores': []})
        candidate_nodes[hit['id']]['semantic_score'] = hit['score']
        
        # Get the structural vector of the pivot node
        struct_vec_of_pivot = meta['structural_vectors'][hit['idx']].reshape(1, -1)
        
        # Search for structurally similar nodes
        D_struct, I_struct = structural_index.search(struct_vec_of_pivot, top_k)
        
        for score, idx in zip(D_struct[0], I_struct[0]):
            if idx >= 0:
                node_id = meta["ids"][idx]
                if node_id not in candidate_nodes:
                     candidate_nodes[node_id] = {'semantic_score': 0, 'structural_scores': []}
                candidate_nodes[node_id]['structural_scores'].append(score)

    return candidate_nodes

def expand_candidates(pivot_hits, structural_index, meta, top_k=settings.TOP_K):
    """
    Expands from pivot nodes to find structurally similar candidates.
    """

    # If you haven't re-indexed yet, this line will crash, forcing you to do Step 1.
    if "struct_ids" not in meta:
        raise KeyError("The metadata file is missing 'struct_ids'. Please re-run the index builder.")
        
    structural_node_ids = meta["struct_ids"]

    # FIX 2: Map node_id to index using the correct list
    # This ensures 'i' corresponds perfectly to the rows in 'structural_vectors'
    node_id_to_struct_idx = { nid: i for i, nid in enumerate(structural_node_ids) }

    candidate_nodes = {}
    
    for hit in pivot_hits:
        node_id = hit['id']

        candidate_nodes[node_id] = candidate_nodes.get(
            node_id,
            {'semantic_score': 0, 'structural_scores': []}
        )
        candidate_nodes[node_id]['semantic_score'] = hit['score']
        
        # FIX 3: Check existence before access
        # (A node might exist in text embeddings but fail structural embedding generation?)
        if node_id in node_id_to_struct_idx:
            struct_idx = node_id_to_struct_idx[node_id]
            
            # Now safe: struct_idx is within bounds [0, 49]
            struct_vec_of_pivot = meta['structural_vectors'][struct_idx].reshape(1, -1)
            
            D_struct, I_struct = structural_index.search(struct_vec_of_pivot, top_k)
            
            for score, idx in zip(D_struct[0], I_struct[0]):
                if idx >= 0:
                    # FIX 4: Resolve neighbor ID using the STRUCTURAL list
                    neighbor_id = structural_node_ids[idx]
                    
                    if neighbor_id not in candidate_nodes:
                        candidate_nodes[neighbor_id] = {
                            'semantic_score': 0,
                            'structural_scores': []
                        }
                    candidate_nodes[neighbor_id]['structural_scores'].append(score)

    return candidate_nodes