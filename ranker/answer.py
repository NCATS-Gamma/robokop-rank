'''
Answer class
'''

import warnings
import datetime
import logging

logger = logging.getLogger(__name__)

class Answerset():
    '''
    An "answer" to a Question.
    Contains a ranked list of walks through the Knowledge Graph.
    '''

    def __init__(self, *args, **kwargs):
        self.answers = []
        self.misc_info = None
        self.filename = None
        self.__idx = 0
        self.timestamp = datetime.datetime.now() # answer creation time

        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, struct[key])
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, kwargs[key])
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

    def toJSON(self):
        keys = [k for k in vars(self) if k[0] is not '_']
        struct = {key:getattr(self, key) for key in keys}
        if 'answers' in struct:
            struct['answers'] = [a.toJSON() for a in struct['answers']]
        if 'timestamp' in struct:
            struct['timestamp'] = struct['timestamp'].isoformat()
        return struct
    
    def toStandard(self):
        '''
        context
        datetime
        id
        message
        original_question_text
        response_code
        result_list
        '''
        json = self.toJSON()
        natural_question = json['misc_info']['natural_question']
        output = {
            'datetime': json['timestamp'],
            'id': '',
            'message': f"{self.misc_info['num_total_paths']} answers found. {len(self.answers)} returned.",
            'response_code': 'OK' if self.answers else 'EMPTY',
            'result_list': [a.toStandard() for a in self.answers]
        }
        return output

    def add(self, answer):
        '''
        Add an Answer to the AnswerSet
        '''

        if not isinstance(answer, Answer):
            raise ValueError("Only Answers may be added to AnswerSets.")

        self.answers += [answer]
        return self

    def __iadd__(self, answer):
        return self.add(answer)

    def __getitem__(self, key):
        return self.answers[key]
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.__idx >= len(self.answers):
            raise StopIteration
        else:
            self.__idx += 1
            return self.answers[self.__idx-1]

    def len(self):
        return len(self.answers)

class Answer():
    '''
    Represents a single answer walk
    '''

    def __init__(self, *args, **kwargs):
        # initialize all attributes
        self.id = None # int
        self.answerset = None # AnswerSet
        self.natural_answer = None # str
        self.nodes = [] # list of str
        self.edges = [] # list of str
        self.score = None # float

        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, struct[key])
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, kwargs[key])
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

    def toJSON(self):
        keys = [k for k in vars(self) if k[0] is not '_']
        struct = {key:getattr(self, key) for key in keys}
        return struct

    def toStandard(self):
        '''
        confidence
        id
        result_graph:
            edge_list:
                confidence
                origin_list
                source_id
                target_id
                type
            node_list:
                accession
                description
                id
                name
                node_attributes
                symbol
                type
        result_type
        text
        '''
        json = self.toJSON()
        output = {
            'confidence': json['score']['rank_score'],
            'id': json['id'],
            'result_graph': {
                'node_list': [standardize_node(n) for n in json['nodes']],
                'edge_list': [standardize_edge(e) for e in json['edges']]
            },
            'result_type': 'individual query answer',
            'text': generate_summary(json['nodes'], json['edges'])
        }
        return output

def generate_summary(nodes, edges):
    # assume that the first node is at one end
    summary = nodes[0]['name']
    latest_node_id = nodes[0]['id']
    node_ids = [n['id'] for n in nodes]
    edges = [e for e in edges if not e['type'] == 'literature_co-occurrence']
    edge_starts = [e['source_id'] for e in edges]
    edge_ends = [e['target_id'] for e in edges]
    while True:
        if latest_node_id in edge_starts:
            idx = edge_starts.index(latest_node_id)
            edge_starts.pop(idx)
            edge_ends.pop(idx)
            edge = edges.pop(idx)
            latest_node_id = edge['target_id']
            latest_node = nodes[node_ids.index(latest_node_id)]
            summary += f" -{edge['type']}-> {latest_node['name']}"
        elif latest_node_id in edge_ends:
            idx = edge_ends.index(latest_node_id)
            edge_starts.pop(idx)
            edge_ends.pop(idx)
            edge = edges.pop(idx)
            latest_node_id = edge['source_id']
            latest_node = nodes[node_ids.index(latest_node_id)]
            summary += f" <-{edge['type']}- {latest_node['name']}"
        else:
            break
    return summary

def standardize_edge(edge):
    '''
    confidence
    provided_by
    source_id
    target_id
    type
    '''
    output = {
        'confidence': edge['weight'],
        'provided_by': edge['edge_source'],
        'source_id': edge['source_id'],
        'target_id': edge['target_id'],
        'type': edge['type'],
        'publications': edge['publications']
    }
    return output

def standardize_node(node):
    '''
    description
    id
    name
    node_attributes
    symbol
    type
    '''
    output = {
        'description': node['name'],
        'id': node['id'],
        'name': node['name'],
        'type': node['type']
    }
    return output