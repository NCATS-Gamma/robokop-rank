from ranker.api.setup import swagger
from ranker.api.util import FromDictMixin

@swagger.definition('Response')
class Response():
    """
    Response
    ---
    type: "object"
    properties:
      type:
        type: "string"
        example: "medical_translator_query_response"
        description: "Entity type of this response"
      id:
        type: "string"
        example: "http://rtx.ncats.io/api/rtx/v1/response/1234"
        description: "URI for this response"
      tool_version:
        type: "string"
        example: "RTX 0.5.1"
        description: "Version label of the tool that generated this response"
      schema_version:
        type: "string"
        example: "0.7.0"
        description: "Version label of this JSON-LD schema"
      datetime:
        type: "string"
        example: "2018-01-09 12:34:45"
        description: "ISO standard datetime string for the time that this response was generated"
      original_question_text:
        type: "string"
        example: "what proteins are affected by sickle cell anemia"
        description: "The original question text typed in by the user"
      restated_question_text:
        type: "string"
        example: "Which proteins are affected by sickle cell anemia?"
        description: "A precise restatement of the question, as understood by the Translator, for which the answer applies. The user should verify that the restated question matches the intent of their original question (it might not)."
      query_type_id:
        type: "string"
        example: "Q2"
        description: "The query type id if one is known for the query/response (as defined in https://docs.google.com/spreadsheets/d/18zW81wteUfOn3rFRVG0z8mW-ecNhdsfD_6s73ETJnUw/edit#gid=1742835901 )"
      terms:
        type: "object"
        example: "{ 'disease': 'malaria' }"
        description: "The is string of the query type id if one is known for the query/response"
      response_code:
        type: "string"
        example: "OK"
        description: "Set to OK for success, or some other short string to indicate and error (e.g., KGUnavailable, TermNotFound, etc.)"
      message:
        type: "string"
        example: "1 answer found"
        description: "Extended message denoting the success or mode of failure for the response"
      result_list:
        type: "array"
        items:
          $ref: "#/definitions/Result"
    """
    pass

@swagger.definition('Result')
class Result():
    """
    Result
    ---
    type: "object"
    description: "One of potentially several results or answers for a query"
    properties:
      id:
        type: "string"
        example: "http://rtx.ncats.io/api/rtx/v1/result/2345"
        description: "URI for this response"
      text:
        type: "string"
        example: "The genetic condition sickle cell anemia may provide protection\
          \ from cerebral malaria via genetic alterations of proteins HBB (P68871)\
          \ and HMOX1 (P09601)."
        description: "A free text description or comment from the reasoner about this answer"
      confidence:
        type: "number"
        format: "float"
        example: 0.9234
        description: "Confidence metric for this result, a value 0.0 (no confidence) and 1.0 (highest confidence)"
      result_type:
        type: "string"
        example: "answer"
        description: "One of several possible result types: 'individual query answer', 'neighborhood graph', 'type summary graph'"
      result_graph:
        $ref: "#/definitions/Result_graph"
    """
    pass

@swagger.definition('Result_graph')
class Result_graph():
    """
    Result_graph
    ---
    type: "object"
    description: "A thought graph associated with this result. This will commonly be a linear path subgraph from one concept to another, but related items aside of the path may be included."
    properties:
      node_list:
        type: "array"
        items:
          $ref: "#/definitions/Node"
      edge_list:
        type: "array"
        items:
          $ref: "#/definitions/ResultEdge"
    """
    pass

@swagger.definition('Query')
class Query():
    """
    Query
    ---
    type: "object"
    required:
      - query_type_id
      - terms
    properties:
      original_question:
        type: "string"
        example: "What proteins are targeted by ibuprofen?"
        description: "Original question as it was typed in by the user"
      query_type_id:
        type: "string"
        example: "Q3"
        description: "RTX identifier for the specific query type"
      max_results:
        type: "integer"
        example: "100"
        description: "Maximum number of individual results to return"
      terms:
        type: "object"
        description: "Dict of terms needed by the specific query type"
        properties:
          chemical_substance:
            type: "string"
            example: "CHEBI:5855"
        additionalProperties: true
    """
    pass
    
@swagger.definition('Node')
class Node(FromDictMixin):
    """
    Node Object
    ---
    id: Node
    required:
        - id
    properties:
        id:
            type: string
            required: true
        type:
            type: string
        name:
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

@swagger.definition('ResultEdge')
class ResultEdge():
    """
    type:
        type: "string"
        example: "affects"
        description: "Higher-level relationship type of this edge"
    relation:
        type: "string"
        example: "affects"
        description: "Lower-level relationship type of this edge"
    source_id:
        type: "string"
        example: "http://omim.org/entry/603903"
        description: "Corresponds to the @id of source node of this edge"
    target_id:
        type: "string"
        example: "https://www.uniprot.org/uniprot/P00738"
        description: "Corresponds to the @id of target node of this edge"
    is_defined_by:
        type: "string"
        example: "RTX"
        description: "A CURIE/URI for the translator group that made the KG"
    provided_by:
        type: "string"
        example: "OMIM"
        description: "A CURIE/URI for the knowledge source that defined this edge"
    confidence:
        type: "number"
        format: "float"
        example: 0.99
        description: "Confidence metric for this edge, a value 0.0 (no confidence) and 1.0 (highest confidence)"
    publications:
        type: "string"
        example: "PubMed:12345562"
        description: "A CURIE/URI for publications associated with this edge"
    evidence_type:
        type: "string"
        example: "ECO:0000220"
        description: "A CURIE/URI for class of evidence supporting the statement made in an edge - typically a class from the ECO ontology"
    qualifiers:
        type: "string"
        example: "ECO:0000220"
        description: "Terms representing qualifiers that modify or qualify the meaning of the statement made in an edge"
    negated:
        type: "boolean"
        example: "true"
        description: "Boolean that if set to true, indicates the edge statement is negated i.e. is not true"
    """
    pass

@swagger.definition('Edge')
class Edge(FromDictMixin):
    """
    Edge Object
    ---
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