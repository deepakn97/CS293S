from collections import defaultdict
import json
from typing import List, Union
from rank_bm25 import BM25Okapi
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet data
nltk.download('stopwords')
nltk.download('wordnet')

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Initialize wordnet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text_bm25(text: str) -> List(str):
    """
    Preprocesses the given text by tokenizing, lemmatizing, and removing stopwords and non-alphabetic tokens.

    :param text: The text to be preprocessed.

    :return: A list of final lemmas after preprocessing.
    """
    tokenized_doc = nlp(text)
    
    # Lemmatize text
    lemmas = [token.lemma_ for token in tokenized_doc]
    
    # Remove stopwords and non-alphabetic tokens
    a_lemmas = [lemma for lemma in lemmas 
                if lemma.isalpha() and lemma not in stopwords.words('english')]
    
    # Lemmatize with NLTK to handle words that spaCy's lemmatizer didn't convert
    final_lemmas = [lemmatizer.lemmatize(a_lemma) for a_lemma in a_lemmas]
    
    return final_lemmas

#TODO: write function to load corpus in chunks
#TODO: write another function to aggregate the chunks blah blah, you get the point. If you don't, skill issue.
def get_bm25_rankings(
    queries: List(dict), 
    corpus: List(dict), 
    N: int = 30
) -> List(str):
    """
    Stores a list of top N document Ids for each query in a dictionary for easy task access.
    Important: check that the ground truth document is always in the top N results.

    :param queries: The queries to be ranked.
    :param corpus: The corpus of documents.
    :param N: The number of top documents to be returned.

    :return: IDs of the top N documents for each query
    """

    # Get the text of the documents
    # Preprocess the documents and pair them with their IDs
    preprocessed_docs = [(doc_id, preprocess_text_bm25(doc['text'])) for doc_id, doc in corpus.items()]

    # Create a BM25 object and initialize it with the preprocessed documents
    bm25 = BM25Okapi([doc_text for doc_id, doc_text in preprocessed_docs])
    
    # Preprocess the queries
    preprocessed_queries = [preprocess_text_bm25(query['query']) for query in queries]

    # For each preprocessed query, get the top N documents
    for i, preprocessed_query in enumerate(preprocessed_queries):
        scores = bm25.get_scores(preprocessed_query)
        top_n_indices = bm25.get_top_n(preprocessed_query, list(range(len(preprocessed_docs))), n=N)
        top_n_ids = [preprocessed_docs[index][0] for index in top_n_indices]
        print(f"Top {N} document IDs for query '{queries[i]['query']}': {top_n_ids}")
    
    return top_n_ids


def compute_metrics(
    ground_truth_ids: List(str), 
    predicted_ids: List(List(str))
) -> dict(str, float):
    """
    Computes Hit@1, Hit@5, Hit@10, MRR and MR.

    :param ground_truth_ids: List of ground truth document IDs for queries.
    :param predicted_ids: The predicted document IDs for each query.

    :return: list of metrics for each query Hit@1, Hit@5, Hit@10, MRR and MR.
    """
    # Initialize dict to store the metrics for each query
    metrics = defaultdict(float)

    # Iterate over ground truth ID for each query
    for i, ground_truth_id in enumerate(ground_truth_ids):
        # If the ground truth ID is in the predicted IDs for the query
        if ground_truth_id in predicted_ids[i]:
            # Get the index of the ground truth ID in the predicted IDs
            index = predicted_ids[i].index(ground_truth_id)
            # Add the rank (index + 1) to the MR list
            metrics['mr'] += index + 1
            # Add the reciprocal rank (1 / (index + 1)) to the MRR list
            metrics['mrr'] += 1 / (index + 1)
            # If the index is less than 10, add a hit to the Hit@10 list
            if index < 10:
                metrics['hit@10'] += 1
                # If the index is less than 5, add a hit to the Hit@5 list
                if index < 5:
                    metrics['hit@5'] += 1
                    # If the index is less than 1, add a hit to the Hit@1 list
                    if index < 1:
                        metrics['hit@1'] += 1
        else:
            # If the ground truth ID is not in the predicted IDs, it's a miss for Hit@1, Hit@5, and Hit@10
            # don't add anything to the MRR, add a fixed value to the MR.
            metrics['mr'] += len(predicted_ids[i]) + 1

    # Calculate the average of the metrics
    for k, v in metrics.items():
        metrics[k] = v / len(ground_truth_ids)

    # Return metrics
    return metrics
        

