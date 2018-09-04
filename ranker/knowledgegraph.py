import time
import os
import sys
import logging
from neo4j.v1 import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)

class KnowledgeGraph:

    def __init__(self):
        # connect to neo4j database
        self.driver = GraphDatabase.driver("bolt://"+os.environ["NEO4J_HOST"]+":"+os.environ["NEO4J_BOLT_PORT"], auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"]))
        self.session = self.driver.session()
        logger.debug('Connected to neo4j.')

    def get_map_for_type(self, type):
        result = self.session.run(f"MATCH (n:{type}) WHERE NOT 'Concept' IN labels(n) AND NOT 'Type' in labels(n) RETURN n")
        records = list(result)
        ids = [r['n'].properties['id'] for r in records]
        synsets = [r['n'].properties['equivalent_identifiers'] for r in records]
        return {syn:id for id, synset in zip(ids,synsets) for syn in synset}

    def similarity_search(self, type1, identifier, type2, by_type, threshhold, maxresults):
        cypher = f"""MATCH (query:{type1} {{id:"{identifier}"}})--(b:{by_type})--(result:{type2}) 
                    WITH query, result, count(distinct b) as intersection, collect(distinct b.id) as i
                    MATCH (query)--(qm:{by_type})
                    WITH query,result, intersection,i, COLLECT(distinct qm.id) AS s1
                    MATCH (result)--(rm:{by_type})
                    WITH query,result,intersection,i, s1, COLLECT(distinct rm.id) AS s2
                    WITH query,result,intersection,s1+filter(x IN s2 WHERE NOT x IN s1) AS union, s1, s2
                    WITH query,result,intersection,union,s1,s2, ((1.0*intersection)/SIZE(union)) AS jaccard
                    WHERE jaccard > {threshhold}
                    RETURN result, jaccard ORDER BY jaccard DESC LIMIT {maxresults}"""
        try:
            logger.debug('hi')
            result = self.session.run(cypher)
            records = list(result)
            retres = [{'id':r['result'].properties['id'],
                       'name':r['result'].properties['name'],
                       'similarity':r['jaccard']} for r in records]
            logger.debug(len(retres))
            retres = list(filter(lambda x: x['id'] != identifier,retres))
            logger.debug(len(retres))
            return retres
        except Exception as e:
            logger.error(e)
            return {},500

    def query(self, question):
        if isinstance(question, str):
            query_string = question
        else:
            query_string = question.cypher(self)

        logger.debug('Running query... ')
        start = time.time()
        result = self.session.run(query_string)
        records = [{'nodes': r['nodes'], 'edges': r['edges']} for r in result]
        logger.debug(f"{time.time()-start} seconds elapsed")

        logger.debug(f"{len(records)} subgraphs returned.")

        return records

    def __del__(self):
        self.session.close()
        logger.debug('Disconnected from database.')
