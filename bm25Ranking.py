"""
This module implements the BM25 ranking algorithm for information retrieval.

The module provides a function to preprocess text by tokenizing, lemmatizing, and removing stopwords and non-alphabetic tokens.
It also provides a function to get BM25 rankings for a given set of queries and a corpus.

The BM25 ranking function stores a list of top N document Ids for each query in a dictionary for easy task access.
It also checks that the ground truth document is always in the top N results.
"""

import os
import json
import nltk
import spacy
import argparse
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from collections import defaultdict
from typing import Dict, List, Union
from nltk.stem import WordNetLemmatizer


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.

    :return argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='BM25 Ranker')
    parser.add_argument('--data', type=str, required=True, help='Path to the data directory.')
    parser.add_argument('--split', type=str, required=True, help='Split of dataset to use.')
    parser.add_argument('--N', type=int, default=30, help='Number of top documents to store/return for each query.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')

    parser.add_argument('--save_ranking', action='store_true', help='Whether to save the ranking.')
    parser.add_argument('--output', type=str, required=False, help='Path to the output file.')

    return parser.parse_args()


class BM25Ranker:
    """
    A class to handle BM25 ranking.
    """
    def __init__(self, corpus: List[Dict]):
        """
        Initializes the BM25Ranker class.

        :param corpus (List[Dict]): The corpus of documents to be ranked.
        """

        # load spacy and nltk resources
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()

        # preprocess the documents and pair them with their IDs
        self.docs = [(doc['id'], self._preprocess_text_bm25(doc['text'])) for doc in tqdm(corpus, total=len(corpus), desc='Preprocessing documents')]

        print('Building BM25 index...')
        self.bm25 = BM25Okapi([doc_text for doc_id, doc_text in self.docs])
        

    def _preprocess_text_bm25(self, text: str) -> List[str]:
        """
        Preprocesses the given text by tokenizing, lemmatizing, and removing stopwords and non-alphabetic tokens.

        :param text (str): The text to be preprocessed.

        :return List[str]: A list of final lemmas after preprocessing.
        """
        tokenized_text = self.nlp(text)
    
        # Lemmatize text
        lemmas = [token.lemma_ for token in tokenized_text]
    
        # Remove stopwords and non-alphabetic tokens
        a_lemmas = [lemma for lemma in lemmas 
                    if lemma.isalpha() and lemma not in stopwords.words('english')]
    
        # Lemmatize with NLTK to handle words that spaCy's lemmatizer didn't convert
        final_lemmas = [self.lemmatizer.lemmatize(a_lemma) for a_lemma in a_lemmas]
    
        return final_lemmas


    def get_rankings(
        self,
        queries: List[Dict], 
        N: int = 30
    ) -> List[str]:
        """
        Stores a list of top N document Ids for each query in a dictionary for easy task access.
        Important: check that the ground truth document is always in the top N results.

        :param queries (List[dict]): The queries to be ranked.
        :param N (int): The number of top documents to be returned.

        :return List[str]: IDs of the top N documents for each query
        """
        # Get the text of the documents
        # Preprocess the documents and pair them with their IDs
        
        # Get the query text
        queries_text = [query['query'] for query in queries]
        
        # Preprocess the queries
        queries_preprocessed = [self._preprocess_text_bm25(query) for query in queries_text]
        
        # Get the top N documents for each query
        top_docs = [self.bm25.get_top_n(query, self.docs, n=N) for query in queries_preprocessed]

        # Return the IDs of the top documents
        return [[doc[0] for doc in query_top_docs] for query_top_docs in top_docs]


def main():
    # load the documents
    print('Loading document map ...')
    with open(f'{args.data}/{args.split}_document_map.json', 'r') as f:
        document_map = json.load(f)

    documents = []
    for doc_id, filename in tqdm(document_map.items(), total=len(document_map), desc='Loading documents'):
        with open(filename, 'r') as f:
            document = json.load(f)
            document['id'] = doc_id
            documents.append(document)

    # load the queries
    print('Loading queries ...')
    with open(f'{args.data}/{args.split}_queries.jsonl', 'r') as f:
        queries = [json.loads(line) for line in f]

    if args.debug:
        queries = queries[:10]
        documents = documents[:100]
        # print('*'*12)
        # print("document: ", documents[0].keys())

    # Instantiate the BM25Ranking class
    bm25_ranker = BM25Ranker(documents)

    # Get the rankings
    rankings = bm25_ranker.get_rankings(queries)

    # Print the rankings
    if args.debug:
        for query, ranking in zip(queries, rankings):
            print(f"Query: {query['query']}")
            print(f"Top documents: " , ranking)

    # Store the rankings
    if args.save_ranking:
        print(f'Saving rankings to {args.output} ...')
        with open(args.output, 'w') as f:
            for query, ranking in zip(queries, rankings):
                f.write(json.dumps({'query_id': query['query_id'], 'bm25': ranking}) + '\n')


if __name__ == "__main__":
    args = parse_args()
    main()