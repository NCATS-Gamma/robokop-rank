"""Core ranker functions."""
from collections import defaultdict
import logging
import os
import requests

logger = logging.getLogger(__name__)


def run(message_json, max_connectivity=-1, max_results=-1, use_novelty=True):
    """Run message through messenger micro-services."""
    plan = [
        ('normalize', {}),
        ('answer', {
            'max_connectivity': max_connectivity,
        }),
        ('yank', {}),
    ]
    if use_novelty:
        plan += [
            ('weight_novelty', {})
        ]
    plan += [
        ('support', {}),
        ('weight_correctness', {}),
        ('score', {}),
        ('screen', {
            'max_results': max_results,
        }),
    ]
    for action, options in plan:
        logger.debug('Calling /%s...', action)
        response = requests.post(
            f"http://{os.environ['MESSENGER_HOST']}:{os.environ['MESSENGER_PORT']}/{action}",
            json={
                'message': message_json,
                'options': options
            }
        )
        if response.status_code != 200:
            raise RuntimeError(response.content)
        message_json = response.json()
    return message_json


def dense(message_json):
    """Convert message to dense result set."""
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
        for nb in result.pop('node_bindings'):
            subgraph['nodes'].append(knode_map[nb['kg_id']])
        for eb in result.pop('edge_bindings'):
            kedge = kedge_map[eb['kg_id']]
            if kedge['id'] == 'literature_co-occurrence':
                continue
            subgraph['edges'].append(kedge)
        subgraph.update(result)
        subgraphs.append(subgraph)
    return subgraphs


def strip_kg(message_json):
    """Strip knowledge graph from message."""
    return {
        'query_graph': message_json['query_graph'],
        'results': message_json['results'],
        'knowledge_graph': None,
    }


def to_robokop(message_json):
    """Convert to ROBOKOP manager expected message form.

    query_graph -> question_graph
    results -> answers
    kg_id/qg_id -> map
    """
    answers = []
    for result in message_json['results']:
        node_bindings = defaultdict(list)
        for nb in result['node_bindings']:
            node_bindings[nb['qg_id']].append(nb['kg_id'])
        edge_bindings = defaultdict(list)
        for eb in result['edge_bindings']:
            edge_bindings[eb['qg_id']].append(eb['kg_id'])
        answers.append({
            'node_bindings': node_bindings,
            'edge_bindings': edge_bindings,
            'score': result['score'],
        })
    output = {
        'question_graph': message_json['query_graph'],
        'knowledge_graph': message_json['knowledge_graph'],
        'answers': answers,
    }
    return output


def csv(message_json):
    """Convert results to CSV string."""
    lines = [[qnode['id'] for qnode in message_json['query_graph']['nodes']]]
    results = message_json['results']
    for result in results:
        nb_map = defaultdict(list)
        for nb in result['node_bindings']:
            nb_map[nb['qg_id']].append(nb['kg_id'])
        lines.append([
            '|'.join(nb_map[qnode_id])
            for qnode_id in lines[0]
        ])
    lines = [','.join(line) for line in lines]
    return '\n'.join(lines)
