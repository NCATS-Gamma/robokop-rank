"""Omnicorp support module."""
import logging
import time
from ranker.support.omnicorp_postgres import OmniCorp

logger = logging.getLogger(__name__)

COUNT_KEY = 'omnicorp_article_count'


def get_supporter():
    """Return an omnicorp support object."""
    # greent should be a greent.core.GreenT
    return OmnicorpSupport()


class OmnicorpSupport():
    """Omnicorp support object."""

    def __init__(self):
        """Create omnicorp support object."""
        self.omnicorp = OmniCorp()

    def term_to_term(self, node_a, node_b):
        """Get number of articles related to both terms and return the result."""
        articles = self.omnicorp.get_shared_pmids(node_a, node_b)
        # count_a = 0
        # count_b = 0
        # if COUNT_KEY in node_a.properties:
        #     count_a = int(node_a.properties[COUNT_KEY])
        # if COUNT_KEY in node_b.properties:
        #     count_b = int(node_b.properties[COUNT_KEY])
        # if (count_a > 0) and (count_b > 0):
        #     articles = self.omnicorp.get_shared_pmids(node_a, node_b)
        # else:
        #     articles = []
        logger.debug(f'OmniCorp {node_a} {node_b} -> {len(articles)}')
        return articles
        # Dont' put these edges into neo4j, just return the article list
        # Even if articles = [], we want to make an edge for the cache.
        # We can decide later to write it or not.

    def get_node_info(self, node):
        """Get node info."""
        count = self.omnicorp.count_pmids(node)
        return {COUNT_KEY: count}

    def prepare(self, nodes):
        """Get list of good nodes?"""
        goodnodes = list(filter(lambda n: self.omnicorp.get_omni_identifier(n) is not None, nodes))
        return goodnodes
