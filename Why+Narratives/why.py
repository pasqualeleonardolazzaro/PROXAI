from neo4j import GraphDatabase
from code_interpreter import ChatBot

uri = "bolt://localhost:7687" 
user = "neo4j"
password = "adminadmin" 

# Create the Connection Driver
driver = GraphDatabase.driver(uri, auth=(user, password))

def get_derivation_column(column_id, activity_id):
    query = (
        "MATCH (c:Column)<-[:WAS_DERIVED_FROM]-(n:Column)-[:WAS_GENERATED_BY]->(a:Activity)-[:WAS_INVALIDATED_BY]-(c) "
        "WHERE n.id = $column_id AND a.id = $activity_id "
        "RETURN c"
    )
    with driver.session() as session:
        result = session.run(query, column_id=column_id, activity_id=activity_id)
        columns = [dict(record["c"]) for record in result]
        if not columns: return None
        return columns[0]

def get_column_and_activities_with_relations(column_id):
    query = (
            "MATCH (n:Column)-[r]-(a:Activity) "
            "WHERE n.id = $column_id "
            "RETURN n, type(r) as relation, a"
        )
    with driver.session() as session:
        result = session.run(query, column_id=column_id)
        results = []
        for record in result:
            node = record["n"]
            relation = record["relation"]
            activity = record["a"]
            results.append((node, relation, activity))
        return results
    

def get_derivation_entity(entity_id, activity_id):
    query = (
        "MATCH (c:Entity)<-[:WAS_DERIVED_FROM]-(n:Entity)-[:WAS_GENERATED_BY]->(a:Activity) "
        "WHERE n.id = $entity_id AND a.id = $activity_id "
        "RETURN c"
    )
    with driver.session() as session:
        result = session.run(query, entity_id=entity_id, activity_id=activity_id)
        entities = [dict(record["c"]) for record in result]
        if not entities: return None
        return entities[0]

def get_entity_and_activities_with_relations(ent_id):
    query = (
            "MATCH (n:Entity)-[r]-(a:Activity) "
            "WHERE n.id = $ent_id "
            "RETURN n, type(r) as relation, a"
        )
    with driver.session() as session:
        result = session.run(query, ent_id=ent_id)
        results = []
        for record in result:
            node = record["n"]
            relation = record["relation"]
            activity = record["a"]
            results.append((node, relation, activity))
        return results


def close_driver():
    driver.close()



node_id = "column:3ba84c28-498d-4ba8-a97b-6729ca203343"
#chat creation for explenations
key = 'your key'
chatbot = ChatBot(api_key=key)
if 'column' in node_id:
    results = get_column_and_activities_with_relations(node_id)
else:
    results = get_entity_and_activities_with_relations(node_id)
for node, relation, activity in results:
    a = dict(activity)
    c = dict(node)
    if relation == "WAS_GENERATED_BY":
        if 'column' in node_id:
            d = get_derivation_column(node_id, a["id"])
        else:
            d = get_derivation_entity(node_id, a["id"])
        expl = chatbot.ask_question(context=a["code"], input=c, type=relation, output=d, element="entitiy")
    else: 
        expl = chatbot.ask_question(context=a["code"], input=c, type=relation)
    print(relation)
    print(a["function_name"])
    print(expl)
    print("\n")

# Closing Driver at the end
close_driver()