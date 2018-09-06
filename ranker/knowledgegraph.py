import time
import os
import sys
import logging
from neo4j.v1 import GraphDatabase, basic_auth
import ranker.api.logging_config
from scipy.stats import hypergeom
from operator import itemgetter

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
                    RETURN result, jaccard ORDER BY jaccard DESC """
        if maxresults > 0:
            cypher += f"""LIMIT {maxresults}"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
            records = list(result)
            retres = [{'id':r['result'].properties['id'],
                       'name':r['result'].properties['name'],
                       'similarity':r['jaccard']} for r in records]
            retres = list(filter(lambda x: x['id'] != identifier,retres))
            return retres
        except Exception as e:
            logger.error(e)
            return {},500

    def get_total_from_cypher(self,node_type_1,node_type_2=None):
        """Get the total count of nodes of type 1, or if node_type_2 is given, the total
        number of type1's that are connected to type 2."""
        if node_type_2 is None:
            cypher = f"""MATCH (a:{node_type_1}) RETURN COUNT(DISTINCT a) as n"""
        else:
            cypher = f"""MATCH (a:{node_type_1})--(p:{node_type_2}) RETURN COUNT(DISTINCT a) as n"""
        with self.driver.session() as session:
            result = list(session.run(cypher))[0]['n']
        return result

    def get_number_of_connected(self,node_type_1,identifiers,node_type_2):
        """Given a set of nodes of type1, count how many of them are connected to a node of type 2"""
        cypher = f"""MATCH (a:{node_type_1})--(p:{node_type_2}) 
                     WHERE a.id IN [{identifiers}] 
                     RETURN COUNT(DISTINCT a) as n"""
        with self.driver.session() as session:
            result = list(session.run(cypher))[0]['n']
        return result

    def enrichment_search(self, identifiers, type1, type2, threshhold, maxresults, num_type1 = None):
        """Given a list of identifiers of type 1, find the elements of type 2 that are connected to the
        query identifiers, and rank them by their enrichment.  A (maximum) p-value threshold can be set,
        as well as the maximum number of results to return.
        To calculate enrichment, the total number of possible type1 entities is needed.  if num_type1 is
        not specified, then the cypher is queried to estimate it, but this estimate may be poor.  If a value is
        passed in, then it is used instead."""
        ids = ','.join([f"'{x}'" for x in identifiers])
        cypher = f"""MATCH (fa:{type1})--(p:{type2})--(d:{type1}) 
                     WHERE fa.id IN [{ids}] AND NOT d.id  in [{ids}] 
                     RETURN p.id as id,p.name as name,COUNT( DISTINCT d ) AS gmx, COUNT(DISTINCT fa) AS x """
        retres = []
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
            # The identifiers that we're taking in are often expanded by bringing in descendants.  However,
            # these descendants are not as well annotated, and many of them are not connected to any phenotypes
            # These don't really count, so remove them.
            T = self.get_number_of_connected(type1,ids,type2)
            if num_type1 is None:
                #Similarly, we want to get the total number of type1, but really only type1's that are connected
                # to ANY type2 should count.
                R = self.get_total_from_cypher(type1,type2)
            else:
                R = num_type1
            for record in  list(result):
                result_id   = record['id']
                result_name = record['name']
                G_minus_x   = record['gmx']
                x           = record['x']
                G = G_minus_x + x
                p = hypergeom.sf(x-1,R,G,T)
                retres.append( {'id':result_id, 'name':result_name, 'p':p } )
            retres = list(filter(lambda x: x['p'] < threshhold,retres))
            retres.sort(key=itemgetter('p'))
            if len(retres) > maxresults and maxresults > 0:
                retres = retres[:maxresults]
            return retres
        except Exception as e:
            logger.error(e)
            return {},500


    def query(self, question, options=None):
        if isinstance(question, str):
            query_string = question
        else:
            query_string = question.cypher(self, options=options)

        logger.debug('Running query... ')
        logger.debug(query_string)
        start = time.time()
        with self.driver.session() as session:
            result = session.run(query_string)
        records = [{'nodes': r['nodes'], 'edges': r['edges']} for r in result]
        logger.debug(f"{time.time()-start} seconds elapsed")

        logger.debug(f"{len(records)} subgraphs returned.")

        return records
