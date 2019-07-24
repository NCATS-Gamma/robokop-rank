"""Search Neo4j for nodes by name.

CALL db.index.fulltext.createNodeIndex("node_name_index", ["named_thing"], ["name"])
"""
import logging
import os
from neo4j.v1 import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)


def get_nodes_by_name(name):
    """Search for nodes by id."""
    terms = name.split(' ')
    patterns = [f"/.*{term}.*/" for term in terms]
    statement = f"""CALL db.index.fulltext.queryNodes('node_name_index', '{
        ' AND '.join(patterns)
    }') YIELD node, score RETURN node.id as curie, node.name as name"""
    logger.debug(statement)
    driver = GraphDatabase.driver(
        f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
        auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(statement)
    driver.close()

    return [dict(record) for record in result]


def count_connections(curie, to_type=None):
    """Count connections to curie.

    Optionally count only connections to a particular type of node.
    """
    if to_type is not None:
        target_spec = ':' + to_type
    else:
        target_spec = ''
    statement = f"""MATCH (n:named_thing {{id:'{curie}'}})
    RETURN size( (n)--({target_spec}) ) as count"""
    logger.debug(statement)
    driver = GraphDatabase.driver(
        f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
        auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(statement)
    driver.close()

    return next(record['count'] for record in result)
