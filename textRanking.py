import os
import argparse
import json
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TextRanking:
    def __init__(self, previous_ranking: List[Dict[str, Any]]):
        print("Loading model ...")
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

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

    
    def get_scores(self, query: List[str], documents: Dict[str, List[str]], ranking_dict: Dict[str, List[str]]) -> torch.Tensor:
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
        tokenized_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

        # Get the last hidden states for the query and documents
        outputs = self.model(**tokenized_input)
        embeddings = self.average_pool(outputs.last_hidden_state, tokenized_input['attention_mask'])

        # Get the query and document embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[0] @ embeddings[1:].T) * 100

        return scores.tolist()
    

    def get_rankings(self, queries: List[Dict[str, Any]],  documents: Dict[str, List[str]], ranking_dict: Dict[str, List[str]]) -> List[List[str]]:
        """
        Get the rankings for the queries

        :param queries (List[dict]): The queries to be ranked.
        :param documents (Dict): The documents to be ranked.

        :return List[str]: IDs of the top N documents for each query
        """

        # Get the scores for the documents
        scores = [self.get_scores(query, documents, ranking_dict) for query in queries]

        # Rerank the documents
        rankings = []
        for query, score in zip(queries, scores):
            ranking = [doc_id for _, doc_id in sorted(zip(score, ranking_dict[query['query_id']]), reverse=True)]
            rankings.append(ranking)

        return rankings



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
    # with open(f'{args.data}/{args.split}_ranking_bm25.jsonl', 'r') as f:
    with open(f'./data/wikiweb2m/ranking_bm25.jsonl', 'r') as f:
        ranking = [json.loads(line) for line in f]

    ranking_dict = {}
    for rank in ranking:
        ranking_dict[rank['query_id']] = rank['bm25']


    if args.debug:
        queries = queries[:10]
        print('*'*12)
        print("document: ", documents[ranking[0]['bm25'][0]].keys())

    # Instantiate the TextRanking class
    text_ranker = TextRanking(ranking)

    # Get the new rankings
    rankings = text_ranker.get_rankings(queries, documents, ranking_dict)

    # Print the rankings
    if args.debug:
        for query, ranking in zip(queries, rankings):
            print(f"Query_id: {query['query_id']}, Query: {query['query']}")
            print(f"Top documents: " , ranking)
    




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BM25 Ranker')
    parser.add_argument('--data', type=str, default="/share/edc/home/dnathani/CS293S/data/wikiweb2m/", help='Path to the data directory.')
    parser.add_argument('--split', type=str, default='test', help='Split of dataset to use.')
    parser.add_argument('--N', type=int, default=30, help='Number of top documents to store/return for each query.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    args = parser.parse_args()

    main(args)



