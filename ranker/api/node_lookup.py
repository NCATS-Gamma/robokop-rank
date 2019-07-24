"""Search Neo4j for nodes by name.

CALL db.index.fulltext.createNodeIndex("node_name_index", ["named_thing"], ["name"])
"""
import logging
import os
from neo4j.v1 import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)


def get_nodes_by_name(name):
    """Search for nodes by id."""
    statement = f"""CALL db.index.fulltext.queryNodes('node_name_index', '{
        ' AND '.join(name.split(' '))
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
