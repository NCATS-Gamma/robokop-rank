"""Definitions following Translator Interchange API specification."""

import copy
from ranker.api.setup import swagger
from ranker.api.util import FromDictMixin2


@swagger.definition('QNode')
class QNode(FromDictMixin2):
    """
    Minimal node specification
    ---
    required:
        - id
    properties:
        id:
            type: string
            description: "Internal ID of this node"
        type:
            type: string
            description: "Optional biolink-model type of this node"
        curie:
            type: string
            description: "Optional curie of this node"
    """

    def __init__(self, *args, **kwargs):
        """Initialize QNode."""
        self.id = None
        self.type = None
        self.curie = None

        super().__init__(*args, **kwargs)


@swagger.definition('QEdge')
class QEdge(FromDictMixin2):
    """
    Minimal edge specification
    ---
    type: object
    required:
        - id
        - source_id
        - target_id
    properties:
        id:
            type: string
            description: "Internal ID of this edge"
        type:
            type: string
            description: "Optional relationship type of this edge"
        source_id:
            type: string
            description: "Internal ID of source node of this edge"
        target_id:
            type: string
            description: "Internal ID of target node of this edge"
    """

    def __init__(self, *args, **kwargs):
        """Initialize QEdge."""
        self.id = None
        self.type = None
        self.source_id = None
        self.target_id = None

        super().__init__(*args, **kwargs)


@swagger.definition('QGraph')
class QGraph(FromDictMixin2):
    """
    Graph representing the minimal question specification
    ---
    type: object
    properties:
        nodes:
            type: array
            items:
                $ref: "#/definitions/QNode"
        edges:
            type: array
            items:
                $ref: "#/definitions/QEdge"
    """

    constructors = {
        'nodes': QNode,
        'edges': QEdge
    }

    def __init__(self, *args, **kwargs):
        """Initialize QGraph."""
        self.nodes = []
        self.edges = []

        super().__init__(*args, **kwargs)

    def apply(self, kmap):
        """Apply a KMap to this QGraph."""
        # add curies to mapped nodes
        nodes = copy.deepcopy(self.nodes)
        for node in nodes:
            if node.id in kmap:
                node.curie = kmap[node.id]
        # remove mapped edges
        edges = [edge for edge in self.edges if edge.id not in kmap]
        return {'machine_question': {
            'nodes': [n.dump() for n in nodes],
            'edges': [e.dump() for e in edges]
        }}


@swagger.definition('KNode')
class KNode(FromDictMixin2):
    """
    Node in the knowledge graph
    ---
    type: object
    required:
        - id
    properties:
        id:
            description: "CURIE identifier for this node"
            type: string
        uri:
            description: "URI identifier for this node"
            type: string
        name:
            description: "Formal name of the entity"
            type: string
        type:
            description: "Entity type of this node (e.g., protein, disease, etc.)"
            type: string
        description:
            description: "One to three sentences of description/definition of this entity"
            type: string
        symbol:
            description: "Short abbreviation or symbol for this entity"
            type: string
        node_attributes:
            description: "A list of arbitrary attributes for the node"
            type: array
            items:
                $ref: "#/definitions/KNode_attribute"
    additionalProperties: true
    """

    def __init__(self, *args, **kwargs):
        """Initialize KNode."""
        self.id = None
        self.type = None
        self.name = None

        super().__init__(*args, **kwargs)

    def __hash__(self):
        """Compute hash for node.

        It's just the id.
        """
        return hash(self.id)

    def __eq__(self, other):
        """Compare node ids."""
        return self.id == other.id


@swagger.definition('KNode_attribute')
class KNode_attribute(FromDictMixin2):
    """
    Generic attribute for a node
    ---
    type: object
    properties:
        type:
            description: "Entity type of this attribute"
            type: string
        name:
            description: "Formal name of the attribute"
            type: string
        value:
            description: "Value of the attribute"
            type: string
        url:
            description: "A URL corresponding to this attribute"
            type: string
    additionalProperties: true
    """

    pass


@swagger.definition('KEdge')
class KEdge(FromDictMixin2):
    """
    Edge in the knowledge graph
    ---
    type: object
    required:
        - id
        - type
        - source_id
        - target_id
    properties:
        id:
            description: "Unique ID for this edge"
            type: string
        type:
            type: string
            description: "Higher-level relationship type of this edge"
        relation:
            type: string
            description: "Lower-level relationship type of this edge"
        source_id:
            type: string
            description: "Corresponds to the @id of source node of this edge"
        target_id:
            type: string
            description: "Corresponds to the @id of target node of this edge"
        is_defined_by:
            type: string
            description: "A CURIE/URI for the translator group that made the KG"
        provided_by:
            type: string
            description: "A CURIE/URI for the knowledge source that defined this edge"
        confidence:
            type: number
            format: float
            description: "Confidence metric for this edge, a value 0.0 (no confidence) and 1.0 (highest confidence)"
        publications:
            type: string
            description: "A CURIE/URI for publications associated with this edge"
        evidence_type:
            type: string
            example: "ECO:0000220"
            description: "A CURIE/URI for class of evidence supporting the statement made in an edge - typically a class from the ECO ontology"
        qualifiers:
            type: string
            description: "Terms representing qualifiers that modify or qualify the meaning of the statement made in an edge"
        negated:
            type: boolean
            description: "Boolean that if set to true, indicates the edge statement is negated i.e. is not true"
    """

    def __init__(self, *args, **kwargs):
        """Initialize KEdge."""
        self.id = None
        self.type = None
        self.source_id = None
        self.target_id = None

        super().__init__(*args, **kwargs)

    def __hash__(self):
        """Compute hash for edge.

        It's just the id.
        """
        return hash(self.id)

    def __eq__(self, other):
        """Compare edge ids."""
        return self.id == other.id


@swagger.definition('KGraph')
class KGraph(FromDictMixin2):
    """
    Graph representing knowledge relevant to a specific question
    ---
    type: object
    properties:
        nodes:
            type: array
            items:
                $ref: "#/definitions/KNode"
        edges:
            type: array
            items:
                $ref: "#/definitions/KEdge"
    """

    constructors = {
        'nodes': KNode,
        'edges': KEdge
    }

    def __init__(self, *args, **kwargs):
        """Initialize KGraph."""
        self.nodes = []
        self.edges = []

        super().__init__(*args, **kwargs)

    def merge(self, other):
        """Merge two KGraphs."""
        self.nodes = list(set(self.nodes + other.nodes))
        self.edges = list(set(self.edges + other.edges))


@swagger.definition('KMap')
class KMap():
    """
    Map from question node and edge IDs to knowledge-graph entity identifiers and relationship references
    ---
    type: object
    additionalProperties:
      oneOf:
        - type: string
        - type: array
          items:
            type: string
    """

    pass


@swagger.definition('Options')
class Options():
    """
    Operation-/module-specific options
    ---
    type: object
    additionalProperties: true
    """

    pass


@swagger.definition('RemoteKGraph')
class RemoteKGraph():
    """
    Pointer to remote knowledge graph
    ---
    type: object
    required:
      - url
    properties:
      url:
        type: string
      credentials:
        $ref: '#/definitions/Credentials'
    """

    pass


@swagger.definition('Credentials')
class Credentials():
    """
    Credentials
    ---
    type: object
    required:
      - username
      - password
    properties:
      username:
        type: string
      password:
        type: string
    """

    pass


@swagger.definition('Message')
class Message(FromDictMixin2):
    """
    Message passed from one module to the next
    ---
    type: object
    required:
      - question_graph
      - knowledge_graph
      - knowledge_maps
    properties:
      question_graph:
        $ref: '#/definitions/QGraph'
      knowledge_graph:
        oneOf:
          - $ref: '#/definitions/KGraph'
          - $ref: '#/definitions/RemoteKGraph'
      knowledge_maps:
        type: array
        items:
          $ref: '#/definitions/KMap'
      options:
        $ref: '#/definitions/Options'
    example:
      question_graph:
        nodes:
          - id: "n00"
            type: "disease"
          - id: "n01"
            type: "gene"
          - id: "n02"
            type: genetic_condition
        edges:
          - id: "e00"
            source_id: "n00"
            target_id: "n01"
          - id: "e01"
            source_id: "n01"
            target_id: "n02"
      knowledge_graph:
        nodes:
          - id: "MONDO:0005737"
            name: "Ebola hemorrhagic fever"
            type: "disease"
        edges: []
      knowledge_maps:
        - n00: "MONDO:0005737"
    """

    constructors = {
        'question_graph': QGraph,
        'knowledge_graph': KGraph
    }

    def __init__(self, *args, **kwargs):
        """Initialize Message."""
        self.question_graph = None
        self.knowledge_graph = None
        self.knowledge_maps = []

        super().__init__(*args, **kwargs)
