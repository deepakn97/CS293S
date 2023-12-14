import os
import json
import math
import argparse
from collections import defaultdict
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from textRanking import TextRanker
from textImageRanking import TextImageRanker



def combine_scores(text_scores, image_scores):
    """
    Combines the text and image scores.

    :param text_scores (Dict[str, List[float]]): The text scores.
    :param image_scores (Dict[str, List[float]]): The image scores.

    :return Dict[str, List[float]]: The combined scores.
    """
    combined_scores = defaultdict(dict)
    # find common query_ids
    common_query_ids = set(text_scores.keys()) & set(image_scores.keys())
    print(f"Number of common query_ids: {len(common_query_ids)}")
    for query_id in common_query_ids:
        score_t = text_scores[query_id]
        score_i = image_scores[query_id]
        # print the keys that are not present in both
        print(set(score_t.keys()) ^ set(score_i.keys()))
        assert len(score_t) == len(score_i), "Length of text and image scores should be the same."
        # check if image score is NaN
        for doc_id in score_t.keys():
            # if no image, just use text score
            if score_i[doc_id] is None:
                combined_scores[query_id] = score_t
            # else, combine the image and text scores
            else:
                combined_scores[query_id] = {doc_id: sum([score_i[doc_id], score_t[doc_id]]) for doc_id in score_i.keys()}
        return combined_scores

def get_rankings(scores):
    """
    calculate the ranking given a dict of scores -> {query_id: {doc_id: score}}
    """

    rankings = {}
    for query_id in scores.keys():
        scores_query = scores[query_id]
        ranking = sorted(scores_query, key=scores_query.get, reverse=True)
        rankings[query_id] = ranking
    return rankings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=None, help='Path to the text data directory.')
    parser.add_argument('--image', type=str, default=None, help='Path to the image data directory.')
    args = parser.parse_args()

    # if args.text is None or args.image is None:
    #     text_scores_path = "./data/wikiweb2m/test_text_scores_30.json"
    #     image_scores_path = "./data/wikiweb2m/test_image_scores_30.json"
    # else:
    text_scores_path = args.text
    image_scores_path = args.image

    # load the text scores
    if args.text is not None:
        print("Loading text scores ...")
        with open(text_scores_path, 'r') as f:
            text_scores = json.load(f)
    
    image_data = []
    if args.image is not None:
        # load the image scores
        print("Loading image scores ...")
        with open(image_scores_path, 'r') as f:
            file_content = f.read().replace('NaN', 'null')
            image_scores = json.loads(file_content)

    # if args.n_samples is not None:
    #     text_scores = {k: v for k, v in text_scores.items() if k in range(args.n_samples)}
    #     image_scores = {k: v for k, v in image_scores.items() if k in range(args.n_samples)}

    if args.text is not None and args.image is not None:
        # combine the scores
        print("Combining scores ...")
        combined_scores = combine_scores(text_scores, image_scores)
    else:
        combined_scores = text_scores if args.text is not None else image_scores

    # save rankings and scores
    filename = f"{os.path.basename(text_scores_path).split('.')[0]}_image_{args.image is not None}_text_{args.text is not None}"
    dirname = os.path.dirname(text_scores_path)
    scores_path = f"{dirname}/{filename}_scores.json"
    rankings_path = f"{dirname}/{filename}_rankings.json"
    with open(scores_path, 'w') as f:
        json.dump(combined_scores, f)
        print(f"Saved scores to {scores_path} ...")
    # with open(rankings_path, 'w') as f:
    #     json.dump(rankings, f)
    # print("Done.")


if __name__ == '__main__':
    main()

