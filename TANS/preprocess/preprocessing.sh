#!/bin/bash

# Step 1: Preprocess the data

# This step is time-consuming, so we run it separately
# You can download the preprocessed data from the following link:
# https://drive.google.com/drive/folders/1iRyGcVZz9xTKErOleT13EtTv3QQC9QN-?usp=sharing

# python node_property.py

# Step 2: Generate text descriptions

for data in cora pubmed; do
    for setting in text_limit text_rich; do
        python citation_graphs.py --data_name $data --setting $setting --without_neigh
        python citation_graphs.py --data_name $data --setting $setting
    done
done

for data in usa brazil europe; do
    python airport_graphs.py --data_name $data --setting text_free
done

echo "Generate text descriptions completed!"