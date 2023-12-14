import os
import argparse
import json
import numpy as np


def evaluate_rankings(rankings):
    """
    evaluate the rankings
    """
    # check that the ground_truth document is no.1
    accuracy = []
    for query_id in rankings.keys():
        ranking = rankings[query_id]
        # remove query_ from query_id
        if isinstance(ranking, dict):
            # sort keys
            ranking = sorted(ranking, key=ranking.get, reverse=True)
        query_id = query_id.replace("query_", "")
        if query_id == ranking[0]:
            accuracy.append(1)
        else:
            print(f"wrong, querry_no: {query_id}")
            accuracy.append(0)
    accuracy = np.mean(accuracy)
    print("accuracy: ", accuracy)
    return accuracy


def compare_rankings(rankings1, rankings2):
    """
    Compare the rankings of each query for two different score rankings.
    Return the queries where one is correct and the other isn't, and where both are wrong.
    """
    common_queries = set(rankings1.keys()).intersection(set(rankings2.keys()))
    ranking1_correct = []
    ranking2_correct = []
    both_wrong = []
    for query_id in common_queries:
        ranking1 = rankings1[query_id]
        ranking2 = rankings2[query_id]

        if isinstance(ranking1, dict):
            ranking1 = sorted(ranking1, key=ranking1.get, reverse=True)
        if isinstance(ranking2, dict):
            ranking2 = sorted(ranking2, key=ranking2.get, reverse=True)

        query_id = query_id.replace("query_", "")

        if query_id == ranking1[0] and query_id != ranking2[0]:
            ranking1_correct.append(query_id)
        elif query_id != ranking1[0] and query_id == ranking2[0]:
            ranking2_correct.append(query_id)
        elif query_id != ranking1[0] and query_id != ranking2[0]:
            both_wrong.append(query_id)
    
    return ranking1_correct, ranking2_correct, both_wrong



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to the data directory.')
    parser.add_argument('--querries', type=str, default=None, help='Path to the querries.')
    args = parser.parse_args()

    if args.data is None:
        # rankings_path = "./data/wikiweb2m/test_rankings_30.json"
        rankings_path = "./data/wikiweb2m/ranking_bm25_1000_10000.jsonl"
    else:
        rankings_path = args.data
    
    # if file ends with .jsonl, convert to dict
    if rankings_path.endswith(".jsonl"):
        print("Converting jsonl to dict ...")
        with open(rankings_path, 'r') as f:
            rankings = [json.loads(line) for line in f]
        rankings_dict = {}
        for rank in rankings:
            rankings_dict[rank['query_id']] = rank['bm25']
        rankings = rankings_dict
        print("Done.")
    else:
        # load the rankings
        print("Loading rankings ...")
        with open(rankings_path, 'r') as f:
            rankings = json.load(f)
    
    # filter by querries (because we haven't encoded all the images yet TODO)
    if args.querries is not None:
        print(f"Loading querries from {args.querries} ...")
        with open(args.querries, 'r') as f:
            querry_keys = json.load(f)
        rankings = {k: v for k, v in rankings.items() if k in querry_keys}
    

    # evaluate the rankings
    print(f"Evaluating rankings for {rankings_path} ...")
    accuracy = evaluate_rankings(rankings)
