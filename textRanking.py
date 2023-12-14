import os
import argparse
import json
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from evaluate import evaluate_rankings

class TextRanker:
    def __init__(self, previous_ranking: List[Dict[str, Any]]):
        print("Loading model ...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)
        self.previous_ranking = previous_ranking

    def average_pool(self,
                     last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average Pool the last hidden states

        :param last_hidden_states(torch.Tensor): The last hidden states of the model
        :param attention_mask(torch.Tensor): The attention mask for the last hidden states
        
        :return torch.Tensor: The average pooled last hidden states
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    
    def get_scores(self, query: List[str], documents: Dict[str, List[str]], ranking_dict: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Get the scores for the documents given the query

        :param query(str): The query to score the documents on
        :param documents(List(str)): The documents to score

        :return torch.Tensor: The scores for the documents
        """
        
        input_text = []
        # Preprocess the query and documents
        input_text.append("query: " + query['query'])

        # For the query, preprocess all texts in each document
        for doc_id in ranking_dict[query['query_id']]:
            input_text.append("passage: " + documents[doc_id]['text'])

        # Tokenize the input
        tokenized_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Get the last hidden states for the query and documents
        outputs = self.model(**tokenized_input)
        embeddings = self.average_pool(outputs.last_hidden_state, tokenized_input['attention_mask'])

        # Get the query and document embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[0] @ embeddings[1:].T) * 100
        scores = {doc_id: score.item() for doc_id, score in zip(ranking_dict[query['query_id']], scores)}

        return scores
    

    def get_rankings(self, queries: List[Dict[str, Any]],  documents: Dict[str, List[str]], ranking_dict: Dict[str, List[str]]) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
        """
        Get the rankings for the queries

        :param queries (List[dict]): The queries to be ranked.
        :param documents (Dict): The documents to be ranked.

        :return List[str]: IDs of the top N documents for each query
        """
        # Get the scores for the documents
        from tqdm import tqdm
        scores = {}
        for query in tqdm(queries):
            scores[query['query_id']] = self.get_scores(query, documents, ranking_dict)
            # Save scores after each step
            with open('text_scores.json', 'w') as f:
                json.dump(scores, f)

        # rerank the documents
        rankings = {}
        for query_id in scores.keys():
            sorted_keys = sorted(scores[query_id], key=lambda k: scores[query_id][k], reverse=True)
            rankings[query_id] = sorted_keys

        return scores, rankings



def main(args):

    # load the documents
    print('Loading document map ...')
    with open(f'{args.data}/{args.split}_document_map.json', 'r') as f:
        document_map = json.load(f)


    documents = {}
    for doc_id, filename in tqdm(document_map.items(), total=len(document_map), desc='Loading documents'):
        with open(filename, 'r') as f:
            document = json.load(f)
            documents[doc_id] = document


    # load the queries
    print('Loading queries ...')
    with open(f'{args.data}/{args.split}_queries.jsonl', 'r') as f:
        queries = [json.loads(line) for line in f]
        print("queries: ", queries[0].keys())    

    # load previous ranking
    print('Loading previous ranking ...')
    with open(args.rankings, 'r') as f:
        ranking = [json.loads(line) for line in f]


    ranking_dict = {}
    for rank in ranking:
        ranking_dict[rank['query_id']] = rank['bm25']


    if args.debug:
        queries = queries[:10]
        print('*'*12)
        print("document: ", documents[ranking[0]['bm25'][0]].keys())
    
    if args.querries is not None:
        print(f"Loading querries from {args.querries} ...")
        with open(args.querries, 'r') as f:
            querry_keys = json.load(f)
        querries = {k: v for k, v in querries.items() if k in querry_keys}

    # Instantiate the TextRanker class
    text_ranker = TextRanker(ranking)

    # Get the new rankings
    scores, rankings = text_ranker.get_rankings(queries, documents, ranking_dict)

    # save scores
    scores_file = f'./data/wikiweb2m/{args.split}_text_scores_{args.N}.json'
    with open(scores_file, 'w') as f:
        json.dump(scores, f)
    print(f'Saved scores to {scores_file} ...')

    # Print the rankings
    if args.debug:
        for query, ranking in zip(queries, rankings):
            print(f"Query_id: {query['query_id']}, Query: {query['query']}")
            print(f"Top documents: " , ranking)
    
    # evaluate rankings
    print('Evaluating rankings ...')
    accuracy = evaluate_rankings(rankings)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BM25 Ranker')
    parser.add_argument('--data', type=str, default="/share/edc/home/dnathani/CS293S/data/wikiweb2m/", help='Path to the data directory.')
    parser.add_argument('--rankings', type=str, default="./data/wikiweb2m/ranking_bm25_full.jsonl", help='Path to the rankings file.')
    parser.add_argument('--split', type=str, default='test', help='Split of dataset to use.')
    parser.add_argument('--N', type=int, default=30, help='Number of top documents to store/return for each query.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    parser.add_argument('--querries', type=str, default=None, help='Path to the querries.')
    args = parser.parse_args()

    main(args)



