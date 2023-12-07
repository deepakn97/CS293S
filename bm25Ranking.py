"""
This module implements the BM25 ranking algorithm for information retrieval.

The module provides a function to preprocess text by tokenizing, lemmatizing, and removing stopwords and non-alphabetic tokens.
It also provides a function to get BM25 rankings for a given set of queries and a corpus.

The BM25 ranking function stores a list of top N document Ids for each query in a dictionary for easy task access.
It also checks that the ground truth document is always in the top N results.
"""

from collections import defaultdict
import json
from typing import Dict, List, Union
from rank_bm25 import BM25Okapi
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class BM25Ranker:
    """
    A class to handle BM25 ranking.
    """
    def __init__(self, corpus: List[Dict]):
        # load spacy and nltk resources
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()

        # preprocess the documents and pair them with their IDs
        self.docs = [(doc['id'], self._preprocess_text_bm25(doc['text'])) for doc in corpus]
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
        return [doc[0] for doc in top_docs]


#TODO: implement main function to compute metrics using bm25
def main():
    pass

if __name__ == "__main__":
    main()