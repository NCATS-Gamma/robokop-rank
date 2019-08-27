import logging
import os
import requests

logger = logging.getLogger(__name__)


def run(message_json):
    plan = [
        'normalize',
        'answer',
        'weight_novelty',
        'yank',
        'support',
        'weight_correctness',
        'score',
    ]
    for action in plan:
        logger.debug('Calling /%s...', action)
        response = requests.post(
            f"http://{os.environ['MESSENGER_HOST']}:{os.environ['MESSENGER_PORT']}/{action}",
            json={
                'message': message_json,
                'options': {}
            }
        )
        if response.status_code != 200:
            raise RuntimeError(response.content)
        message_json = response.json()
    return message_json


def dense(message_json):
    subgraphs = []
    knode_map = {
        knode['id']: knode
        for knode in message_json['knowledge_graph']['nodes']
    }
    kedge_map = {
        kedge['id']: kedge
        for kedge in message_json['knowledge_graph']['edges']
    }
    for result in message_json['results']:
        subgraph = {
            'nodes': [],
            'edges': [],
        }
        for nb in result['node_bindings']:
            subgraph['nodes'].append(knode_map[nb['kg_id']])
        for eb in result['edge_bindings']:
            kedge = kedge_map[eb['kg_id']]
            if kedge['id'] == 'literature_co-occurrence':
                continue
            subgraph['edges'].append(kedge)


def strip_kg(message_json):
    return {
        'query_graph': message_json['query_graph'],
        'results': message_json['results'],
        'knowledge_graph': None,
    }
