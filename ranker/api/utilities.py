"""Ranker utility endpoints."""
import logging
import os
from flask import request
from flask_restful import Resource
from neo4j.v1 import GraphDatabase, basic_auth
from ranker.api.setup import api

logger = logging.getLogger(__name__)


def node_cypher(node_spec):
    """Generate cypher for node specification."""
    statements = []
    statements.append(':' + node_spec.pop('type', 'named_thing'))
    if node_spec:
        statements.append('{' + ', '.join([f'{key}: "{value}"' for key, value in node_spec.items()]) + '}')
    return '(' + ' '.join(statements) + ')'


class CountConnections(Resource):
    def post(self):
        """
        Count connections between biomedical entities.
        ---
        tags: [util]
        requestBody:
            description: node specifications
            content:
                application/json:
                    schema:
                        type: array
                        items:
                            type: object
                            properties:
                                type:
                                    type: string
                                id:
                                    type: string
                        minItems: 2
                        maxItems: 2
                    example:
                      - type: disease
                        id: "MONDO:0005737"
                      - {}
            required: true
        responses:
            200:
                description: Number of connections.
                content:
                    application/json:
                        schema:
                            type: number
        """
        statement = f"""
        MATCH {node_cypher(request.json[0])}-[e]-{node_cypher(request.json[1])}
        RETURN count(e) as n
        """
        logger.debug(statement)

        driver = GraphDatabase.driver(
            f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
            auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
        )
        with driver.session() as session:
            result = session.run(statement)
        driver.close()

        return [
            record['n']
            for record in result
        ][0], 200

api.add_resource(CountConnections, '/count_connections/')


class CountPredicates(Resource):
    def post(self):
        """
        Count predicates between two biomedical entities.
        ---
        tags: [util]
        requestBody:
            description: node specifications
            content:
                application/json:
                    schema:
                        type: array
                        items:
                            type: object
                            properties:
                                type:
                                    type: string
                                id:
                                    type: string
                        minItems: 2
                        maxItems: 2
                    example:
                      - type: disease
                        id: "MONDO:0005737"
                      - type: gene
            required: true
        responses:
            200:
                description: Number of connections.
                content:
                    application/json:
                        schema:
                            type: object
                            additionalProperties:
                                type: number
        """
        statement = f"""
        MATCH {node_cypher(request.json[0])}-[e]->{node_cypher(request.json[1])}
        RETURN DISTINCT type(e) as predicate, count(e) as n
        """
        logger.debug(statement)

        driver = GraphDatabase.driver(
            f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_BOLT_PORT']}",
            auth=basic_auth("neo4j", os.environ["NEO4J_PASSWORD"])
        )
        with driver.session() as session:
            result = session.run(statement)
        driver.close()

        return {
            record['predicate']: record['n']
            for record in result
        }, 200

api.add_resource(CountPredicates, '/count_predicates/')
