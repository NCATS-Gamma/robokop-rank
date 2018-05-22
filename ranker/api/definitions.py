from ranker.api.setup import swagger
from ranker.api.util import FromDictMixin

@swagger.definition('Query')
class Query():
    """
    type: "object"
    properties:
      original_question:
        type: "string"
        example: "what genetic conditions offer protection against malaria"
        description: "Original question as it was typed in by the user"
      restated_question:
        type: "string"
        example: "Which genetic conditions may offer protection against malaria?"
        description: "Restatement of the question as understood by the translator"
      message:
        type: "string"
        example: "Your question was understood."
        description: "Response from the translation engine to the user"
      known_query_type_id:
        type: "string"
        example: "Q1"
        description: "RTX identifier for the specific query type"
      bypass_cache:
        type: "string"
        example: "true"
        description: "Set to true in order to bypass any possible cached response and try to answer the query over again"
      max_results:
        type: "integer"
        example: "100"
        description: "Maximum number of individual results to return"
      page_size:
        type: "integer"
        example: 100
        description: "Split the results into pages with this number of results each"
      page_number:
        type: "integer"
        example: 1
        description: "Page number of results when the number of results exceeds the page_size"
      terms:
        type: "object"
        description: "Dict of terms needed by the specific query type"
        properties:
          disease:
            type: "string"
            example: "malaria"
        additionalProperties: true
    """
    pass
    
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