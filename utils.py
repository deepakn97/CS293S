from collections import defaultdict
from typing import Dict, List

def compute_metrics(
    ground_truth_ids: List[str], 
    predicted_rankings: List[List[str]]
) -> Dict[str, float]:
    """
    Computes Hit@1, Hit@5, Hit@10, MRR and MR.

    :param ground_truth_ids (List[str]): List of ground truth document IDs for queries.
    :param predicted_rankings (List[List[str]]): The predicted document rankings for each query.

    :return Dict[str, float]: Dictionary of metrics for each query Hit@1, Hit@5, Hit@10, MRR and MR.
    """
    # Initialize dict to store the metrics for each query
    metrics = defaultdict(float)

    # Iterate over ground truth ID for each query
    for i, ground_truth_id in enumerate(ground_truth_ids):
        # If the ground truth ID is in the predicted IDs for the query
        if ground_truth_id in predicted_rankings[i]:
            # Get the index of the ground truth ID in the predicted IDs
            index = predicted_rankings[i].index(ground_truth_id)
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
            metrics['mr'] += len(predicted_rankings[i]) + 1

    # Calculate the average of the metrics
    for k, v in metrics.items():
        metrics[k] = v / len(ground_truth_ids)

    # Return metrics
    return metrics