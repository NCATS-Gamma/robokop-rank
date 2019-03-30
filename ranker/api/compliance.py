"""Compliance utilities."""


def message2std(message):
    """Convert ROBOKOP message format to standard message format.

    ROBOKOP: https://github.com/NCATS-Gamma/robokop-rank/blob/master/ranker/definitions.py
    standard: https://github.com/NCATS-Tangerine/NCATS-ReasonerStdAPI/blob/master/API/TranslatorReasonersAPI.yaml
    """
    message['query_graph'] = message.pop('question_graph')
    for node in message['query_graph']['nodes']:
        node['node_id'] = node.pop('id')
    for edge in message['query_graph']['edges']:
        edge['edge_id'] = edge.pop('id')
    return message


def std2message(query):
    """Convert ROBOKOP message format to standard message format.

    ROBOKOP: https://github.com/NCATS-Gamma/robokop-rank/blob/master/ranker/definitions.py
    standard: https://github.com/NCATS-Tangerine/NCATS-ReasonerStdAPI/blob/master/API/TranslatorReasonersAPI.yaml
    """
    message = query['query_message']
    message['question_graph'] = message.pop('query_graph')
    for node in message['question_graph']['nodes']:
        node['id'] = node.pop('node_id')
    for edge in message['question_graph']['edges']:
        edge['id'] = edge.pop('edge_id')
    return message
