'''
Answer class
'''

import warnings
import datetime

class Answerset():
    '''
    An "answer" to a Question.
    Contains a ranked list of walks through the Knowledge Graph.
    '''

    def __init__(self, *args, **kwargs):
        self.answers = []
        self.question_info = None
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
        restated_question_text
        result_code
        result_list
        '''
        json = self.toJSON()
        natural_question = json['question_info']['natural_question']
        output = {
            'context': 'context',
            'datetime': json['timestamp'],
            'id': 'uid',
            'message': f"{len(self.answers)} potential answers found.",
            'original_question_text': natural_question,
            'restated_question_text': f"An improved version of '{natural_question}'?",
            'result_code': 'OK' if self.answers else 'EMPTY',
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
        self.answer_set = None # AnswerSet
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
            'text': 'short answer name here'
        }
        return output

def standardize_edge(edge):
    '''
    confidence
    origin_list
    source_id
    target_id
    type
    '''
    output = {
        'confidence': edge['scoring']['edge_proba'],
        'origin_list': edge['edge_source'],
        'source_id': edge['start'],
        'target_id': edge['end'],
        'type': edge['predicate']
    }
    return output

def standardize_node(node):
    '''
    accession
    description
    id
    name
    node_attributes
    symbol
    type
    '''
    output = {
        'accession': 'accession',
        'description': node['name'],
        'id': node['id'],
        'name': node['id'],
        'node_attributes': None,
        'symbol': None,
        'type': node['type']
    }
    return output