"""
This module houses the TextImageRanker class, a high-level tool for image ranking based on text queries.
It leverages the open_clip library to handle the heavy lifting, including model creation and data preprocessing.
The TextImageRanker class provides methods for image and text preprocessing, as well as extracting image paths from documents.
"""


import ast
from collections import defaultdict
from torchvision.transforms import ToTensor
import torch
import open_clip
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple

class TextImageRanker:
    def __init__(self):
        model, train_preprocessor, val_preprocessor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', jit=False)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor

        print(type(val_preprocessor))
        print(type(self.val_preprocessor))

    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocesses the images in a doc for the CLIP model.

        :param image_paths (List[str]): The paths to the images.
        :return torch.Tensor: The preprocessed images.
        """
        images = [Image.open(image_path) for image_path in image_paths]
        return torch.stack([self.val_preprocessor(image).unsqueeze(0) for image in images]) # type: ignore

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocesses the text for the CLIP model.

        :param text (List[str]): The batch of text to be preprocessed.
        :return torch.Tensor: The preprocessed text.
        """
        return self.tokenizer(text)
    
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
            image_paths[doc['id']] = ast.literal_eval(doc['image_paths'])
        
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
            image_tensors = self.preprocess_images(doc_images)
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors) # type: ignore
                text_features = self.model.encode_text(preprocessed_query) # type: ignore

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

        return sorted_keys

def main():
    text_image_ranker = TextImageRanker()

    # TODO: Look at the textRanking.py file and implement the reranking 


if __name__ == '__main__':
    main()