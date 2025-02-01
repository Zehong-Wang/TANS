#!/bin/bash

enc_model=minilm
llm_model=gpt-4o-mini

for data in cora pubmed; do
    # text_rich
    python encode_text.py --data_name $data --enc_model $enc_model --llm_model $llm_model --node_text title abstract
    python encode_text.py --data_name $data --enc_model $enc_model --llm_model $llm_model --node_text title abstract --wo_neigh

    # text_limit
    python encode_text.py --data_name $data --enc_model $enc_model --llm_model $llm_model --node_text title
    python encode_text.py --data_name $data --enc_model $enc_model --llm_model $llm_model --node_text abstract
done

for data in usa brazil europe; do
    # text_free
    python encode_text.py --data_name $data --enc_model $enc_model --llm_model $llm_model --node_text none
done

echo "Encode text completed!"