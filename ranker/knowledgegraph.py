import time
import os
import sys
import logging
from neo4j.v1 import GraphDatabase, basic_auth
import ranker.api.logging_config

logger = logging.getLogger(__name__)

class KnowledgeGraph:

    def __init__(self):
        # connect to neo4j database
        self.driver = GraphDatabase.driver("bolt://"+os.environ["NEO4J_HOST"]+":"+os.environ["NEO4J_BOLT_PORT"], auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"]))

    def get_map_for_type(self, type):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{type}) WHERE NOT 'Concept' IN labels(n) AND NOT 'Type' in labels(n) RETURN n")
        records = list(result)
        ids = [r['n'].properties['id'] for r in records]
        synsets = [r['n'].properties['equivalent_identifiers'] for r in records]
        return {syn:id for id, synset in zip(ids,synsets) for syn in synset}

    def query(self, question):
        if isinstance(question, str):
            query_string = question
        else:
            query_string = question.cypher(self)

        logger.debug('Running query... ')
        logger.debug(query_string)
        start = time.time()
        with self.driver.session() as session:
            result = session.run(query_string)
        records = [{'nodes': r['nodes'], 'edges': r['edges']} for r in result]
        logger.debug(f"{time.time()-start} seconds elapsed")

        logger.debug(f"{len(records)} subgraphs returned.")

        return records

    def __del__(self):
        self.session.close()
        logger.debug('Disconnected from database.')
