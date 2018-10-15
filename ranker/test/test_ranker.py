import pytest
from ranker.ranker import Ranker

def test_hi():
    ranker = Ranker()
    assert True

def test_one_hop():
    q = { 'nodes': [ {"id": 0, "type": "disease", "curie": "MONDO:0005148"},
                     {"id": 1, "type": "chemical_substance" }],
          'edges': [ {'source_id': 1, 'target_id': 0, 'id':'a'} ]
          }
    g = {
         'nodes': [ {"id": "MONDO:0005148", "type": "disease"},
                    {"id": "CHEBI:17234", "type": "chemical_substance"}],
         'edges': [ {'id': 1, 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:17234', 'type':'contributes_to'},
                    {'id': 'xyz', 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:17234', 'type':'literature_co-occurrence','publications':range(10)}]
    }
    sg = {
        'nodes': {'n0':'MONDO:0005148', 'n1':'CHEBI:17234'},
        'edges': {'ea': [1], 's0': 'xyz'}
    }
    ranker = Ranker(question = q, graph = g)
    #Normally weights calculated in code, but here generating by fiat
    g['edges'][0]['weight'] = 0.1
    g['edges'][1]['weight'] = 1.0
    l,n = ranker.graph_laplacian(sg)
    v = ranker.subgraph_statistic(sg,metric_type='volt')

def test_three_hop():
    q = { 'nodes': [ {"id": 0, "type": "disease", "curie": "MONDO:0005148"},
                     {"id": 1, "type": "chemical_substance" },
                     {"id": 2, "type": "gene" },
                     {"id": 3, "type": "chemical_substance" }],
          'edges': [ {'source_id': 1, 'target_id': 0, 'id':'a'},
                     {'source_id': 2, 'target_id': 1, 'id':'b'},
                     {'source_id': 3, 'target_id': 2, 'id':'c'} ]
          }
    g = {
         'nodes': [ {"id": "MONDO:0005148", "type": "disease"},
                    {"id": "CHEBI:17234", "type": "chemical_substance"},
                    {"id": "HGNC:601", "type": "gene"},
                    {"id": "CHEBI:123", "type": "chemical_substance"}
                    ],
         'edges': [ {'id': 1, 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:17234', 'type':'contributes_to'},
                    {'id': 2, 'source_id': 'HGNC:601', 'target_id': 'CHEBI:17234', 'type':'increases_response_to'},
                    {'id': 3, 'target_id': 'HGNC:601', 'source_id': 'CHEBI:123', 'type':'increases_secretion_of'},
                    {'id': '1to2', 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:17234', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': '1to3', 'target_id': 'MONDO:0005148', 'source_id': 'HGNC:601', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': '1to4', 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:123', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': '2to3', 'target_id': 'HGNC:601', 'source_id': 'CHEBI:17234', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': '2to4', 'target_id': 'CHEBI:123', 'source_id': 'CHEBI:17234', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': '3to4', 'target_id': 'HGNC:601', 'source_id': 'CHEBI:123', 'type':'literature_co-occurrence','publications':range(10)},
                    {'id': 4, 'target_id': 'MONDO:0005148', 'source_id': 'CHEBI:17234', 'type':'contributes_to'}]
    }
    sg = {
        'nodes': {'n0':'MONDO:0005148', 'n1':'CHEBI:17234', 'n2': 'HGNC:601', 'n3': 'CHEBI:123'},
        'edges': {'ea': [1,4], 'eb': [2], 'ec': [3],  's0': '1to2', 's1':'1to3', 's2':'1to4', 's3': '2to3', 's4': '2to4', 's5': '3to4'}
    }
    ranker = Ranker(question = q, graph = g)
    #Normally weights calculated in code, but here generating by fiat
    answer = 0
    if answer == 0:
        g['edges'][0]['weight'] = 0.2   # A to B
        g['edges'][1]['weight'] = 0.42  # B to C
        g['edges'][2]['weight'] = 0.67  # C to D
        g['edges'][3]['weight'] = 0.37  # A to B
        g['edges'][4]['weight'] = 0.55  # A to C
        g['edges'][5]['weight'] = 0.52  # A to D
        g['edges'][6]['weight'] = 0.58  # B to C
        g['edges'][7]['weight'] = 0.57  # B to D
        g['edges'][8]['weight'] = 0.67  # C to D
        g['edges'][9]['weight'] = 0.2   # A to B again
    if answer == 1:
        g['edges'][0]['weight'] = 1.00   # A to B
        g['edges'][1]['weight'] = 1.00  # B to C
        g['edges'][2]['weight'] = 1.00  # C to D
        g['edges'][3]['weight'] = 1.00  # A to B
        g['edges'][4]['weight'] = 1.00  # A to C
        g['edges'][5]['weight'] = 1.00  # A to D
        g['edges'][6]['weight'] = 1.00  # B to C
        g['edges'][7]['weight'] = 1.00  # B to D
        g['edges'][8]['weight'] = 1.00  # D to D



    print('')
    for i in range(10):
        v0 = ranker.subgraph_statistic(sg,metric_type='volt')
        l0,n = ranker.graph_laplacian(sg)
        g['edges'][i]['weight'] += 5
        v1 = ranker.subgraph_statistic(sg,metric_type='volt')
        l1,n = ranker.graph_laplacian(sg)
        g['edges'][i]['weight'] -= 5
        print(i, v0, v1)
    #print(l0)
    #print(l1)

    #Changing B to C doesn't change the score, why?
