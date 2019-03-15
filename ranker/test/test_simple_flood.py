#!/bin/bash

for i in {1..17}
do
   curl -X POST "http://127.0.0.1/api/simple/quick/?rebuild=false&output_format=MESSAGE&max_connectivity=0&max_results=250" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"machine_question\":{\"edges\":[{\"source_id\":\"n0\",\"target_id\":\"n1\"},{\"source_id\":\"n1\",\"target_id\":\"n2\"}],\"nodes\":[{\"curie\":\"MONDO:0005737\",\"id\":\"n0\",\"type\":\"disease\"},{\"id\":\"n1\",\"type\":\"gene\"},{\"id\":\"n2\",\"type\":\"genetic_condition\"}]},\"name\":\"Ebola--(gene)--(genetic_condition)\",\"natural_question\":\"What genetic conditions might provide protection against Ebola?\",\"notes\":\"#ebola #q1\"}" &
done

