import time
import os
import sys
import logging
from neo4j.v1 import GraphDatabase, basic_auth
from ranker.universalgraph import UniversalGraph

logger = logging.getLogger(__name__)

class KnowledgeGraph:

    def __init__(self):
        # connect to neo4j database
        self.driver = GraphDatabase.driver("bolt://"+os.environ["NEO4J_HOST"]+":"+os.environ["NEO4J_BOLT_PORT"], auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"]))
        self.session = self.driver.session()
        logger.debug('Connected to neo4j.')

    def queryToGraph(self, query_string):
        result = list(self.session.run(query_string))
        query_graph = UniversalGraph.record2networkx(result)

        return query_graph

    def get_map_for_type(self, type):
        result = self.session.run(f"MATCH (n:{type}) WHERE NOT 'Concept' IN labels(n) AND NOT 'Type' in labels(n) RETURN n")
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
        start = time.time()
        result = self.session.run(query_string)
        records = [r['nodes'] for r in result]
        logger.debug(f"{time.time()-start} seconds elapsed")

        logger.debug(f"{len(records)} subgraphs returned.")

        return records

    def __del__(self):
        self.session.close()
        logger.debug('Disconnected from database.')
