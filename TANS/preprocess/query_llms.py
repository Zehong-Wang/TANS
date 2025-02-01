import numpy as np
import pandas as pd
import openai
from openai import OpenAI

import argparse
import os.path as osp

api_key = "ADD YOUR API KEY HERE"


def parse_args():
    parser = argparse.ArgumentParser(description='Query LLM models for graph node classification')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name', choices=['cora', 'pubmed', 'usa', 'brazil', 'europe'])
    parser.add_argument('--setting', type=str, default='text_rich', help='Text setting', choices=['text_rich', 'text_limit', 'text_free'])
    parser.add_argument('--without_neigh', action='store_true', help='Exclude neighbor info')
    return parser.parse_args()


def get_file_paths(args):
    """Get paths for question and answer files"""
    base_path = osp.join(osp.dirname(__file__), '../../data/response', args.setting, args.dataset)
    question_file = 'question_wo_neigh.csv' if args.without_neigh else 'question.csv'
    answer_file = f'answer_{args.model}_wo_neigh.csv' if args.without_neigh else f'answer_{args.model}.csv'

    return osp.join(base_path, question_file), osp.join(base_path, answer_file)


def query_model(client, question, model):
    """Query the LLM model"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant in graph learning and graph theory."
                },
                {"role": "user", "content": question}
            ],
            max_completion_tokens=512
        )
        answer = completion.choices[0].message.content
        return answer.replace("\n", " ").replace("  ", " ")
    except:
        return "Error"


def main():
    args = parse_args()
    client = OpenAI(api_key=api_key)

    question_path, answer_path = get_file_paths(args)

    # Load data
    question_df = pd.read_csv(question_path)
    df = pd.read_csv(answer_path, sep="\t")

    # Process rows with errors
    while "Error" in df["answer"].values:
        for i in range(len(df)):
            if df.iloc[i]["answer"] == "Error":
                answer = query_model(client, question_df.iloc[i]["question"], args.model)
                if answer != "Error":
                    print(f'Successfully processed line {i}.')
                df.at[i, "answer"] = answer
                df.to_csv(answer_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
