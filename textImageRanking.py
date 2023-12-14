"""
This module houses the TextImageRanker class, a high-level tool for image ranking based on text queries.
It leverages the open_clip library to handle the heavy lifting, including model creation and data preprocessing.
The TextImageRanker class provides methods for image and text preprocessing, as well as extracting image paths from documents.
"""

import os
import glob
import math
from typing import Any, Dict, List, Tuple
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from PIL import Image
import argparse
from torchvision.transforms import ToTensor
import torch
from multiprocessing import Pool
import multiprocessing as mp
import open_clip
from evaluate import evaluate_rankings

class TextImageRanker:
    def __init__(self, device, ranking_dict: Dict[str, List[str]], documents: Dict[str, Dict[str, Any]]):
        model, train_preprocessor, val_preprocessor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', jit=False)
        self.device = device
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(self.device)
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.ranking_dict = ranking_dict
        self.documents = documents
        self.scores = defaultdict(dict)
        self.rankings = defaultdict(dict)


    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocesses the images in a doc for the CLIP model.

        :param image_paths (List[str]): The paths to the images.
        :return torch.Tensor: The preprocessed images.
        """
        images = [Image.open(image_path) for image_path in image_paths]
        ims = [self.val_preprocessor(image).to(self.device) for image in images]
        return torch.stack(ims) # type: ignore

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocesses the text for the CLIP model.

        :param text (List[str]): The batch of text to be preprocessed.
        :return torch.Tensor: The preprocessed text.
        """
        return self.tokenizer(text).to(self.device)
    
    # using object as type in Dict gives type error later
    def get_image_paths(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Gets the image paths from the document.

        :param doc (Dict[str, Dict[str, object]]): The document.

        :return Dict[str, List[str]]: The image paths.
        """
        image_paths = defaultdict(list)
        for doc_id in documents.keys():
            doc = documents[doc_id]
            image_paths[doc_id] = doc['image_paths']
        
        return image_paths


    def get_scores(self, query: str, documents: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculates the text image similarity score between the given query and each document.

        :param query (str): The text query to be used for reranking.
        :param docs (Dict[str, List[str]]): Document IDs mapped to image paths in the document.

        :return Dict[str, np.float64]: Document IDs mapped to the mean similarity score.
        """

        # Preprocess the queries
        preprocessed_query = self.preprocess_text(query)

        # For each query, preprocess all images in each document and calculate the mean similarity
        document_scores = defaultdict(float)
        for doc_id, doc_images in documents.items():
            if len(doc_images) == 0: # if no image, skip
                # save as NaN
                document_scores[doc_id] = -1
            else:
                image_tensors = self.preprocess_images(doc_images)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensors.to(self.device)) # type: ignore
                    text_features = self.model.encode_text(preprocessed_query.to(self.device)) # type: ignore

                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate the cosine similarity
                similarity = (image_features @ text_features.T).squeeze(0)

                # Calculate the mean similarity for the document
                document_score = torch.mean(similarity)
                document_scores[doc_id] = document_score.item()

        return document_scores

    def get_rankings(self, query: str, documents: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Reranks the documents based on the mean similarity score between document and query.

        :param query (str): The text query to be used for reranking.
        :param document_scores (Dict[str, Dict[str, Any]]): Dictionary of documents.

        :return List[str]: Reranked list of document IDs.
        """

        # get image paths
        image_paths = self.get_image_paths(documents)

        # get scores
        doc_scores = self.get_scores(query, image_paths)

        # Rerank the documents based on the mean similarity score
        sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)        
        
        return doc_scores, sorted_keys

    def rerank_dataset(self, queries: List[Dict[str, Any]], documents: Dict[str, Dict[str, Any]], ranking_dict: Dict[str, List[str]]) -> List[List[str]]:
        """
        iterate over querries and rerank the corresponding documents using
        text-image embeddings.

        :param queries (List[Dict[str, Any]]): The queries to be ranked.
        :param documents (Dict[str, Dict[str, Any]]): The documents to be ranked.
        :param ranking_dict (Dict[str, List[str]]): The previous ranking of the documents.
        """
        # get the rankings and scores
        results = defaultdict(dict)
        scores = {}
        rankings = {}
        for query in tqdm(queries, desc="Processing queries"):
            print(f"Query_id: {query['query_id']}, Query: {query['query']}")
            # obtain the documents for the query
            ranked_docs = {query_id: documents[query_id] for query_id in ranking_dict[query['query_id']]}
            doc_scores, ranking = self.get_rankings(query['query'], ranked_docs)
            scores[query['query_id']] = doc_scores
            rankings[query['query_id']] = ranking
            # Save scores after each step
            with open('./data/wikiweb2m/image_scores.json', 'w') as f:
                json.dump(scores, f)

        return scores, rankings


def process_query(query, ranker):
    print(f"Query_id: {query['query_id']}, Query: {query['query']}")
    file_path = f'./data/wikiweb2m/text_image_scores/image_scores_{query["query_id"]}.json'
    if os.path.exists(file_path):
        return None, None, None
    ranked_docs = {query_id: ranker.documents[query_id] for query_id in ranker.ranking_dict[query['query_id']]}
    doc_scores, ranking = ranker.get_rankings(query['query'], ranked_docs)
    ranker.scores[query['query_id']] = doc_scores
    ranker.rankings[query['query_id']] = ranking
    # Save scores after each step
    with open(file_path, 'w') as f:
        json.dump(ranker.scores, f)
    return query['query_id'], doc_scores, ranking

def rerank_dataset(device, queries, ranking_dict, documents):
    ranker = TextImageRanker(device, ranking_dict, documents)
    with Pool(processes=8) as pool:
        for query_id, doc_scores, ranking in tqdm(pool.starmap(process_query, [(query, ranker) for query in queries]), desc="Processing queries"):
            ranker.scores[query_id] = doc_scores
            ranker.rankings[query_id] = ranking

    

def main():
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
    
    queries = queries[:10000]

    ranking_dict = {}
    for rank in ranking:
        ranking_dict[rank['query_id']] = rank['bm25']

    if args.debug:
        queries = queries[:10]
        print('*'*12)
        print("document: ", documents[ranking[0]['bm25'][0]].keys())
    

    # get the image paths
    print('Loading document images ...')
    images_dir = "./data/wikiweb2m/images"
    images_paths = glob.glob(os.path.join(images_dir, "*.jpg"))

    # Instantiate the TextRanker class
    # text_image_ranker = TextImageRanker()

    # Get the new rankings
    # scores, rankings = text_image_ranker.rerank_dataset(queries, documents, ranking_dict)

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    queries_parts = np.array_split(queries, len(devices))
    for device, queries in zip(devices, queries_parts):
        rerank_dataset(device, queries, ranking_dict, documents)

    # save the scores
    scores_file = f'./data/wikiweb2m/{args.split}_image_scores_{args.N}.json'
    with open(scores_file, 'w') as f:
        json.dump(scores, f)
    print(f'Saved scores to {scores_file}')

    # Print the rankings
    if args.debug:
        for query, ranking in zip(queries, rankings):
            print(f"Query_id: {query['query_id']}, Query: {query['query']}")
            print(f"Top documents: " , ranking)
    
    

if __name__ == '__main__':
    mp.set_start_method('spawn')  # set the start method to 'spawn'
    parser = argparse.ArgumentParser(description='BM25 Ranker')
    parser.add_argument('--data', type=str, default="./data/wikiweb2m", help='Path to the data directory.')
    parser.add_argument('--rankings', type=str, default="./data/wikiweb2m/ranking_bm25.jsonl", help='Path to the rankings file.')
    parser.add_argument('--split', type=str, default='test', help='Split of dataset to use.')
    parser.add_argument('--N', type=int, default=30, help='Number of top documents to store/return for each query.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    args = parser.parse_args()
    
    main()