"""Search Neo4j for nodes by name.

CALL db.index.fulltext.createNodeIndex("node_name_index", ["named_thing"], ["name"])
"""
import logging
import os
import re
from neo4j.v1 import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)


def get_nodes_by_name(name, node_type=None):
    """Search for nodes by id."""
    terms = name.split(' ')
    patterns = ["/.*" + re.sub(r'\\', r'\\\\', re.escape(term)) + ".*/" for term in terms]
    statement = f"""CALL db.index.fulltext.queryNodes('node_name_index', '{
        ' AND '.join(patterns)
    }') YIELD node, score"""
    if node_type:
        statement += f' WHERE "{node_type}" IN labels(node)'
    statement += " RETURN node.id as curie, node.name as name, labels(node) as type, score as search_score, size( (node)--() ) as degree"
    logger.debug(statement)
    driver = GraphDatabase.driver(
        f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
        auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(statement)
    driver.close()

    return [dict(record) for record in result]

def count_connections(curie):
    """Count connections to curie.

    Optionally count only connections to a particular type of node.
    """
    
    statement = f"""MATCH (n:named_thing {{id:'{curie}'}})--(m)
    WITH DISTINCT [label IN labels(m) WHERE label <> "named_thing" | label] AS types, count(m) AS nums
    UNWIND types AS type
    RETURN type, sum(nums) AS num"""
    logger.debug(statement)
    driver = GraphDatabase.driver(
        f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
        auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(statement)
    driver.close()

    return [dict(record) for record in result]
