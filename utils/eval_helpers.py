import numpy as np

def calc_acc_at_top_n(probabilities_with_candidates, gold_labels, top_n):
    """
    Calculate probabilities for 0 top_n-1 top ranked items
    :param probabilities_with_candidates: The probabilities of all candidates
    :param gold_labels: The probabilities of of all candidates
    :param top_n: Top n to calculate accuracy for
    :return: List of accuracies
    """

    probs_as_array = np.asarray(probabilities_with_candidates, dtype=np.float64)
    probs_argsorted = np.argsort(probs_as_array, axis=-1)

    if top_n > probs_argsorted.shape[1]:
        raise Exception("top_n %s can be max %s" % (top_n, probs_argsorted.shape[1]))
    print probs_argsorted
    acc_top = np.zeros(top_n)

    for pi in range(0, probs_argsorted.shape[0]):
        for topi in range(top_n):
            if gold_labels[pi] in probs_argsorted[pi][-(topi+1):]:
                acc_top[topi] += 1.0

    for topi in range(top_n):
        acc_top[topi] = acc_top[topi] / probs_argsorted.shape[0]

    return acc_top

def ensemble_epoch_runs_by_accuracy(epoch_candidate_probs, epoch_accuracy):
    """
    Multiplies each probability by the accuracy for the corresponding epoch.
    Sums the probabilities
    :param epoch_candidate_probs: Probabilities for each epoch
    :param epoch_accuracy: The epoch accuracy
    :return: Returns the summary probability per candidate 
    """
    epoch_candidate_probs_arr = np.asarray(epoch_candidate_probs, dtype=np.float64)

    acc = np.expand_dims(np.expand_dims(100 * np.asarray(epoch_accuracy, dtype=np.float64), -1), -1)

    acc_scaled = acc * epoch_candidate_probs_arr

    return np.sum(np.asarray(acc_scaled), axis=0)

if __name__ == '__main__':
    print "Batch helpers"
    probs = [
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3],
    ]

    gold = [0, 1, 2, 3, 4]

    accuracies = calc_acc_at_top_n(probs, gold, top_n=5)

    print accuracies

