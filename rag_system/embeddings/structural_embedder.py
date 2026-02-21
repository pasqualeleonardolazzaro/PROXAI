import numpy as np
from rag_system.config import settings

def compute_node2vec_embeddings(driver, dim=settings.NODE2VEC_DIM):
    """
    Computes structural embeddings using Neo4j GDS node2vec.
    """
    # Create a GDS in-memory projection
    with driver.session() as session:
        session.run("CALL gds.graph.drop('provGraph', false)")
        session.run("""
        CALL gds.graph.project(
            'provGraph',
            '*',
            '*'
        )
        """)

    # Run node2vec and write results back to Neo4j temporarily
    with driver.session() as session:
        session.run(f"""
        CALL gds.node2vec.write('provGraph', {{
            embeddingDimension: {dim},
            writeProperty: 'node2vec_emb',
            walkLength: {settings.NODE2VEC_WALK_LENGTH},
            iterations: {settings.NODE2VEC_ITERATIONS},
            inOutFactor: {settings.NODE2VEC_IN_OUT_FACTOR},
            returnFactor: {settings.NODE2VEC_RETURN_FACTOR},
            windowSize: {settings.NODE2VEC_WINDOW_SIZE},
            concurrency: 2
        }})
        YIELD nodePropertiesWritten, computeMillis
        """)

    # Extract embeddings into a Python dictionary
    embeddings = {}
    with driver.session() as session:
        result = session.run("MATCH (n) WHERE n.node2vec_emb IS NOT NULL RETURN id(n) as id, n.node2vec_emb as emb")
        for record in result:
            embeddings[record["id"]] = np.array(record["emb"], dtype=np.float32)

    # Clean up by dropping the graph projection and removing the property
    with driver.session() as session:
        session.run("CALL gds.graph.drop('provGraph', false)")
        session.run("MATCH (n) WHERE n.node2vec_emb IS NOT NULL REMOVE n.node2vec_emb")
        print("Cleaned up GDS projection and node2vec_emb property from Neo4j.")

    return embeddings