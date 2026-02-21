
def cypher_expand_context(driver, node_id, hops=1):
    """
    Retrieves the central node and its immediate connections.
    Uses OPTIONAL MATCH to ensure the central node is returned even if isolated.
    """
    try:
        nid = int(node_id)
    except (ValueError, TypeError):
        # handles string ids
        nid = node_id

    cypher = """
    MATCH (n)
    WHERE id(n) = $nid
    
    // Get immediate neighbors (1 hop)
    OPTIONAL MATCH (n)-[r]-(m)
    
    // Return flat rows of (CentralNode, Relationship, Neighbor)
    RETURN 
        labels(n) as n_labels, 
        properties(n) as n_props,
        type(r) as rel_type,
        properties(r) as rel_props,
        startNode(r) = n as is_outgoing,
        labels(m) as m_labels, 
        properties(m) as m_props
    LIMIT 100
    """

    with driver.session() as session:
        result = session.run(cypher, nid=nid)
        rows = list(result)

        if not rows:
            return None

        # Extract Central Node Data (from the first row)
        first_row = rows[0]
        central_node = {
            "labels": first_row["n_labels"],
            "props": dict(first_row["n_props"]) if first_row["n_props"] else {}
        }

        # Extract Relationships
        relationships = []
        for row in rows:
            # If OPTIONAL MATCH found nothing, rel_type will be None
            if row["rel_type"] is None:
                continue
                
            relationships.append({
                "rel_type": row["rel_type"],
                "rel_props": dict(row["rel_props"]) if row["rel_props"] else {},
                "neighbor_labels": row["m_labels"],
                "neighbor_props": dict(row["m_props"]) if row["m_props"] else {},
                "is_outgoing": row["is_outgoing"]
            })

    return {
        "central_node": central_node,
        "relationships": relationships
    }