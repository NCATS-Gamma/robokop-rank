"""Omnicorp service module."""
import datetime
import os
import logging
import psycopg2
from ranker.util import Text

logger = logging.getLogger(__name__)


class OmniCorp():
    """Omnicorp service object."""

    def __init__(self):
        """Create and omnicorp service object."""
        db = os.environ['OMNICORP_DB']
        user = os.environ['OMNICORP_USER']
        port = os.environ['OMNICORP_PORT']
        host = os.environ['OMNICORP_HOST']
        password = os.environ['OMNICORP_PASSWORD']
        self.prefixes = set([
            'UBERON',
            'BSPO',
            'PATO',
            'GO',
            'MONDO',
            'HP',
            'ENVO',
            'OBI',
            'CL',
            'SO',
            'CHEBI',
            'HGNC',
            'MESH'])
        logger.info("Opening Connection to ROBOKOPDB Postgres")
        self.conn = psycopg2.connect(
            dbname=db,
            user=user,
            host=host,
            port=port,
            password=password)
        self.nsingle = 0
        self.total_single_call = datetime.timedelta()
        self.npair = 0
        self.total_pair_call = datetime.timedelta()

    def __del__(self):
        self.conn.close()

    def close(self):
        self.conn.close()

    def get_omni_identifier(self, node_id):
        """Get omnicorp identifier."""
        # Let's start with just the 'best' identifier
        identifier = node_id
        prefix = Text.get_curie(node_id)
        if prefix not in self.prefixes:
            logger.debug(f"What kinda tomfoolery is this?\n" +
                         f"{node_id}")
                         #  f"{node.id} {node.type}\n" +
                         #  f"{node.synonyms}")
            return None
        return identifier

    def get_shared_pmids(self, node1, node2):
        """Get shared PMIDs."""
        id1 = self.get_omni_identifier(node1)
        id2 = self.get_omni_identifier(node2)
        if id1 is None or id2 is None:
            return []
        pmids = self.postgres_get_shared_pmids(id1, id2)
        if pmids is None:
            logger.error("OmniCorp gave up")
            return None
        return [f'PMID:{p}' for p in pmids]

    def get_shared_pmids_count(self, node1, node2):
        """Get shared PMIDs."""
        id1 = self.get_omni_identifier(node1)
        id2 = self.get_omni_identifier(node2)
        if id1 is None or id2 is None:
            return []
        pmid_count = self.postgres_get_shared_pmids_count(id1, id2)
        if pmid_count is None:
            logger.error("OmniCorp gave up")
            return None
        return pmid_count

    def postgres_get_shared_pmids(self, id1, id2):
        """Get shared PMIDs from postgres?"""
        prefix1 = Text.get_curie(id1)
        prefix2 = Text.get_curie(id2)
        start = datetime.datetime.now()
        cur = self.conn.cursor()
        statement = f"SELECT a.pubmedid\n" + \
                    f"FROM omnicorp.{prefix1} a\n" + \
                    f"JOIN omnicorp.{prefix2} b ON a.pubmedid = b.pubmedid\n" + \
                    f"WHERE a.curie = %s\n" + \
                    f"AND b.curie = %s"
        try:
            cur.execute(statement, (id1, id2))
            pmids = [x[0] for x in cur.fetchall()]
            cur.close()
            end = datetime.datetime.now()
            self.total_pair_call += (end-start)
            logger.debug(f"Found {len(pmids)} shared ids in {end-start}\n" +
                        f"Total {self.total_pair_call}")
            self.npair += 1
            if self.npair % 100 == 0:
                logger.info(f"NCalls: {self.npair}\n" +
                            f"Total time: {self.total_pair_call}\n" +
                            f"Avg Time: {self.total_pair_call/self.npair}")
            return pmids
        except psycopg2.ProgrammingError as err:
            self.conn.rollback()
            cur.close()
            logger.debug(f'OmniCorp query error: {str(err)}\nReturning [].')
            return []

    def postgres_get_shared_pmids_count(self, id1, id2):
        """Get shared PMIDs from postgres?"""
        prefix1 = Text.get_curie(id1)
        prefix2 = Text.get_curie(id2)
        cur = self.conn.cursor()
        statement = f"SELECT COUNT(a.pubmedid)\n" + \
                    f"FROM omnicorp.{prefix1} a\n" + \
                    f"JOIN omnicorp.{prefix2} b ON a.pubmedid = b.pubmedid\n" + \
                    f"WHERE a.curie = %s\n" + \
                    f"AND b.curie = %s"
        try:
            cur.execute(statement, (id1, id2))
            pmid_count = cur.fetchall()[0][0]
            cur.close()
            return pmid_count
        except psycopg2.ProgrammingError as err:
            self.conn.rollback()
            cur.close()
            logger.debug(f'OmniCorp query error: {str(err)}\nReturning 0.')
            return 0


    def count_pmids(self, node):
        """Count PMIDs and return result."""
        identifier = self.get_omni_identifier(node)
        if identifier is None:
            return 0
        prefix = Text.get_curie(identifier)
        start = datetime.datetime.now()
        cur = self.conn.cursor()
        statement = f"SELECT COUNT(pubmedid) from omnicorp.{prefix}\n" + \
                    "WHERE curie = %s"
        try:
            cur.execute(statement, (identifier,))
            n = cur.fetchall()[0][0]
            cur.close()
            end = datetime.datetime.now()
            self.total_single_call += (end-start)
            logger.debug(f"""Found {n} pmids in {end-start}
                        Total {self.total_single_call}""")
            self.nsingle += 1
            if self.nsingle % 100 == 0:
                logger.info(f"NCalls: {self.nsingle}\n" +
                            f"Total time: {self.total_single_call}\n" +
                            f"Avg Time: {self.total_single_call/self.nsingle}")
            return n
        except psycopg2.ProgrammingError as err:
            self.conn.rollback()
            cur.close()
            logger.debug(f'OmniCorp query error: {str(err)}\nReturning 0.')
            return 0
