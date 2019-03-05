#!/usr/bin/env python

import os
import json
import requests

with open('./gamma_ebola_p8.json','r') as f8:
    as8 = json.load(f8)

with open('./gamma_qgraph.json', 'r') as fq:
    qgraph = json.load(fq)['question_graph']


post_data = as8
post_data['question_graph'] = qgraph

response = requests.post('http://127.0.0.1:6011/api/normalize/', json=post_data)
print(response)
print(response.text)

message = response.json()

with open('translated_output_api.json','w') as fout:
    json.dump(message, fout, indent=2)
