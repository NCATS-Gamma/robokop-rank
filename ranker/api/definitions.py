from ranker.api.setup import swagger
from ranker.api.util import FromDictMixin

@swagger.definition('Node')
class Node(FromDictMixin):
    """
    Node Object
    ---
    schema:
        id: Node
        required:
            - id
        properties:
            id:
                type: string
                required: true
            type:
                type: string
            identifiers:
                type: array
                items:
                    type: string
                default: []
    """
    def __init__(self, *args, **kwargs):
        self.id = None
        self.type = None
        self.identifiers = []

        super().__init__(*args, **kwargs)

    def dump(self):
        return {**vars(self)}

@swagger.definition('Edge')
class Edge(FromDictMixin):
    """
    Edge Object
    ---
    schema:
        id: Edge
        required:
            - start
            - end
        properties:
            start:
                type: string
            end:
                type: string
            min_length:
                type: integer
                default: 1
            max_length:
                type: integer
                default: 1
    """
    def __init__(self, *args, **kwargs):
        self.start = None
        self.end = None
        self.min_length = 1
        self.max_length = 1

        super().__init__(*args, **kwargs)

    def dump(self):
        return {**vars(self)}

@swagger.definition('Question')
class Question(FromDictMixin):
    """
    Question Object
    ---
    schema:
        id: Question
        required:
          - nodes
          - edges
        properties:
            nodes:
                type: array
                items:
                    $ref: '#/definitions/Node'
            edges:
                type: array
                items:
                    $ref: '#/definitions/Edge'
        example:
            nodes:
              - id: 0
                type: disease
                identifiers: ["MONDO:0005737"]
              - id: 1
                type: gene
              - id: 2
                type: genetic_condition
            edges:
              - start: 0
                end: 1
              - start: 1
                end: 2
    """

    def __init__(self, *args, **kwargs):
        '''
        keyword arguments: id, user, notes, natural_question, nodes, edges
        q = Question(kw0=value, ...)
        q = Question(struct, ...)
        '''
        # initialize all properties
        self.nodes = [] # list of nodes
        self.edges = [] # list of edges

        super().__init__(*args, **kwargs)

    def preprocess(self, key, value):
        if key == 'nodes':
            return [Node(n) for n in value]
        elif key == 'edges':
            return [Edge(e) for e in value]

    def dump(self):
        return {**vars(self)}