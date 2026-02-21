def _get_display_name(props):
    """
    Helper to find a human-readable name from a dictionary of properties.
    """
    # Keys to look for 
    priority_keys = ['function_name','name','value']
    
    # Clean keys to lower case for checking
    prop_map = {k.lower(): k for k in props.keys()}
    
    for key in priority_keys:
        if key in prop_map:
            return str(props[prop_map[key]])
            
    # return the first string property found, or "Unknown"
    for k, v in props.items():
        if isinstance(v, str) and len(v) < 200: # max characters
            return f"{v}"
            
    return "Unnamed Entity"


def _clean_props(props, is_central=False):
    """
    Removes heavy fields.
    - Neighbors: Skip long strings entirely to save space.
    - Central Node: TRUNCATE long strings so we get context but not overflow.
    """
    # Keys to ALWAYS exclude
    blacklist = {'embedding', 'vector', 'embedding_node2vec', 'full_text_embedding'}
    
    # 1. SET LIMITS
    # 600 chars is roughly 150-200 tokens. 
    # If a node has 10 properties, that's max 2000 tokens.
    MAX_CENTRAL_LEN = 600 
    MAX_NEIGHBOR_LEN = 100
    
    cleaned = {}
    for k, v in props.items():
        if k.lower() in blacklist:
            continue
        
        # Check Value Type
        if isinstance(v, str):
            if is_central:
                # SAFEGUARD: If central node text is huge, truncate it.
                if len(v) > MAX_CENTRAL_LEN:
                    cleaned[k] = v[:MAX_CENTRAL_LEN] + "... [truncated]"
                else:
                    cleaned[k] = v
            else:
                # Neighbor logic: Skip entirely if too long (keep graph concise)
                if len(v) > MAX_NEIGHBOR_LEN:
                    continue
                cleaned[k] = v
        else:
            # Keep numbers/booleans as they are usually token-cheap
            cleaned[k] = v
            
    return cleaned

def format_context(expanded_contexts):
    """
    Converts the graph data into a dense, triple-based text format.
    """
    final_text = []

    for ctx in expanded_contexts:
        node_id = ctx['node_id']
        score = ctx['score']
        graph_data = ctx['graph_data']
        
        if not graph_data:
            continue

        # Header for this chunk
        central_node = graph_data['central_node']
        c_props = _clean_props(central_node['props'], is_central=True)
        c_name = _get_display_name(c_props)
        c_labels = ":".join(central_node['labels'])
        
        # Section Header
        chunk_lines = [f"--- Context Centered on: {c_name} ({c_labels}) [Score: {score:.2f}] ---"]
        
        # Add central node properties 
        prop_str = ", ".join([f"{k}: {v}" for k, v in c_props.items()])
        chunk_lines.append(f"Details: {prop_str}")
        chunk_lines.append("Connections:")

        # Format Relationships
        # Format: (Subject) --[RELATION]--> (Object)
        for rel in graph_data['relationships']:
            n_props = _clean_props(rel['neighbor_props'], is_central=False)
            n_name = _get_display_name(n_props)
            n_labels = ":".join(rel['neighbor_labels'])
            
            rel_type = rel['rel_type']
            
            # Add relationship properties if they exist 
            r_props_str = ""
            if rel['rel_props']:
                clean_r_props = _clean_props(rel['rel_props'])
                if clean_r_props:
                    r_props_str = f" ({', '.join([f'{k}:{v}' for k,v in clean_r_props.items()])})"

            # Arrow logic
            if rel['is_outgoing']:
                # (Central) -> (Neighbor)
                line = f"  ({c_name}) --[{rel_type}{r_props_str}]--> ({n_name} : {n_labels})"
            else:
                # (Neighbor) -> (Central)
                line = f"  ({n_name} : {n_labels}) --[{rel_type}{r_props_str}]--> ({c_name})"
            
            chunk_lines.append(line)
        
        final_text.append("\n".join(chunk_lines))

    return "\n\n".join(final_text)

def format_analytic_result(result):
    """
    Formats results including the Cypher query for context.
    """
    lines = ["!!! DATABASE DIRECT QUERY RESULTS !!!"]
    
    # Provide the Query 
    if 'cypher_query' in result:
        lines.append(f"Query executed: {result['cypher_query']}")

    # Handle Facts 
    if result['facts']:
        lines.append(f"Calculated Values: {', '.join(result['facts'])}")
        
    # Handle Nodes
    nodes = result['nodes']
    if nodes:
        lines.append(f"Found {len(nodes)} items matching your query:")
        
        #truncates to avoid context explosion
        limit = 100 
        for i, n in enumerate(nodes):
            if i >= limit:
                lines.append(f"... and {len(nodes) - limit} more items (truncated).")
                break
            # Use safe get
            content_preview = n.get('content', '')[:150].replace("\n", " ") 
            lines.append(f"- {content_preview}")
            
    return "\n".join(lines)