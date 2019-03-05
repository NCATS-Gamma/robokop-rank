#!/usr/bin/env python

import os
import json

with open('./gamma_ebola_p8.json','r') as f8:
    as8 = json.load(f8)

with open('./gamma_qgraph.json', 'r') as fq:
    qgraph = json.load(fq)['question_graph']

# Check Q graph to ensure that there are no ambiguous nodes or edges
qnode_types = {}
qnode_ids = {}
for qn in qgraph['nodes']:
    qn_type = qn['type']
    if qn_type in qnode_types:
        raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes of the same type.')
    else:
        qnode_types[qn_type] = qn['id']
        qnode_ids[qn['id']] = qn_type

# Edges are found by comparing the source_type and target_type
qedge_types = {}
qedge_ids = {}
for qe in qgraph['edges']:
    n_source = [n for n in qgraph['nodes'] if n['id'] == qe['source_id']]
    if len(n_source) > 1:
        raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
    n_source = n_source[0]
    n_target = [n for n in qgraph['nodes'] if n['id'] == qe['target_id']]
    if len(n_target) > 1:
        raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
    n_target = n_target[0]

    qe_types = (n_source['type'], n_target['type'])

    if qe_types in qedge_types:
        raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple edges connecting the same types.')
    else:
        qedge_types[qe_types] = qe['id']
        qedge_ids[qe['id']] = qe_types

# If we made it to this point we know that,
# 1) We only have question nodes with unique types
# 2) We only have question edges with unique sets of source and target node types

edge_id_fun = lambda e: f"({e['source_id']})-({e['type']};{e['provided_by']})-({e['target_id']})"
knowledge_graph = {
    'nodes': [],
    'edges': []
}
answers = []
node_ids = set()
edge_ids = set()
for r in as8['result_list']:
    ans_nodes = r['result_graph']['node_list']
    ans_edges = r['result_graph']['edge_list']

    # Form knowledge_graph nodes
    for n in ans_nodes:
        if n['id'] not in node_ids:
            node_ids.add(n['id'])
            # n['curie'] = n['id']
            knowledge_graph['nodes'].append(n)

    # Form knowledge_graph edge
    for e in ans_edges:
        if 'id' not in e:
            # edge doesn't have an id
            # We have to make up an id
            # Using source_id, target_id, type, provided_by
            e['id'] = edge_id_fun(e)
        
        edge_id = e['id']
        e['source'] = e['provided_by']
        
        if edge_id not in edge_ids:
            edge_ids.add(edge_id)
            knowledge_graph['edges'].append(e)

    # Form new answer
    # We need to figure out which answer node corresponds to which node in the q_graph
    # Similarly for edges
    # Above, we checked the the qgraph to ensure that nodes can be identified by type and
    # edges can be identified by the tuple of (source_type, target_type)
    new_answer = {
        'node_bindings': {},
        'edge_bindings': {},
        'score': []
    }
    
    for qn_id in qnode_ids:
        these_nodes = [n for n in ans_nodes if n['type'] == qnode_ids[qn_id]]
        new_answer['node_bindings'][qn_id] = [n['id'] for n in these_nodes]
    
    edge_id_tuples = {}
    for e in ans_edges:

        n_source = [n for n in ans_nodes if n['id'] == e['source_id']]
        if len(n_source) > 1:
            raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
        n_source = n_source[0]
        n_target = [n for n in ans_nodes if n['id'] == e['target_id']]
        if len(n_target) > 1:
            raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
        n_target = n_target[0]
        
        edge_id_tuples[edge_id_fun(e)] = (n_source['type'], n_target['type'])

    # print(edge_id_tuples)

    for qe_id in qedge_ids:
        these_edges = [e for e in ans_edges if edge_id_tuples[edge_id_fun(e)] == qedge_ids[qe_id]]
        # print(f'{qe_id} - {edge_id_fun(e)} - {these_edges}')
        new_answer['edge_bindings'][qe_id] = [e['id'] for e in these_edges]
    

    new_answer['score'] = r['confidence']
    
    answers.append(new_answer)

message = {
    'question_graph': qgraph,
    'knowledge_graph': knowledge_graph,
    'answers': answers
}

with open('translated_output.json','w') as fout:
    json.dump(message, fout, indent=2)
