from neo4j import GraphDatabase
from rag_system.config import settings

def connect_neo4j():
    """Establishes a connection to the Neo4j database."""
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASS), notifications_disabled_categories=["DEPRECATION"])
    return driver

def fetch_graph(driver):
    """
    Fetches all nodes and relationships from Neo4j.
    """
    with driver.session() as session:
        # Fetch nodes with their internal id, labels, and properties
        nodes_res = session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props")
        nodes = {r["id"]: {"labels": r["labels"], "props": r["props"]} for r in nodes_res}

        # Fetch relationships
        rels_res = session.run("MATCH (a)-[r]->(b) RETURN id(a) as a, id(b) as b, type(r) as t")
        edges = [(r["a"], r["b"], r["t"]) for r in rels_res]
        
    return nodes, edges