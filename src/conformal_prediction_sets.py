# Source code for conformal prediction

import numpy as np
import torch
import torch.nn.functional as F
import random
import os

def seed_everything(seed:int):
    """
    Sets a seed for reproducibility
    :param seed: seed
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Threshold_Conformer:

    def __init__(self, alpha, model, dataset, calibration_mask, seed):

        seed_everything(seed)

        self.alpha = alpha
        self.model = model
        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        logit_scores = self.model(self.x, self.edge_index)[calibration_mask]
        softmax_scores = F.softmax(logit_scores, dim=1)

        uncertainty_scores = 1 - softmax_scores
        true_label_indices = torch.arange(len(y_calibration))
        uncertainty_scores_true_label = uncertainty_scores[true_label_indices, y_calibration]

        threshold_quantile = np.ceil(((n+1)*(1-self.alpha)))/n

        #Calculate the variance to make sure  we are still in the bounds
        self.variance = (self.alpha * (1-self.alpha))/(n+2)

        #Sort the uncertainty scores from low to large
        uncertainty_scores_true_label_sorted = torch.sort(uncertainty_scores_true_label, descending=False)[0]

        return torch.quantile(uncertainty_scores_true_label_sorted, threshold_quantile, interpolation='higher')

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=-1)
        uncertainty_scores = 1 - softmax_scores

        #Get a mask to check which predictions are below the threshold
        uncertainty_scores_below_threshold = uncertainty_scores <= self.threshold_q

        #Get a mask for all predictions that manage to get over the threshold
        #softmax_above_threshold = softmax_scores >= 1 - self.threshold_q

        self.prediction_sets = [torch.nonzero(mask).squeeze(dim=1).tolist() for mask in uncertainty_scores_below_threshold]
        #self.prediction_sets = [torch.nonzero(mask).squeeze(dim=1).tolist() for mask in softmax_above_threshold]

        return self.prediction_sets

class Adaptive_Conformer:

    def __init__(self, alpha, model, dataset, calibration_mask, random_split=False, lambda_penalty=0., k_reg=None, seed=7):

        seed_everything(seed)

        self.alpha = alpha
        self.model = model
        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.lambda_penalty = lambda_penalty
        self.k_reg = k_reg
        self.random_split = random_split

        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        logit_scores = self.model(self.x, self.edge_index)[calibration_mask]
        softmax_scores = F.softmax(logit_scores, dim=1)

        #Get the softmax score of the true class
        softmax_score_true_class = softmax_scores[torch.arange(softmax_scores.shape[0]), y_calibration].reshape(-1, 1)

        # Get the sorted probabilities from large to low
        softmax_scores_sorted, softmax_scores_sorted_indices = torch.sort(softmax_scores, descending=True, dim=1)

        #Check if we have a penalty to add
        if self.lambda_penalty > 0.:
            softmax_scores_sorted = self._get_regularized_softmax_scores(softmax_scores_sorted, softmax_scores_sorted_indices, y_calibration)

        # Now we summed up the probabilities
        cumulative_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        # Get the random score
        if self.random_split:
            u_vec = torch.rand_like(softmax_score_true_class)
            random_noise = u_vec * softmax_score_true_class

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)

        softmax_scores_cut = torch.tensor([cumulative_softmax_scores[sample, idx] for sample, idx in cutoff_idx]).reshape(-1, 1)

        #Add the random noise to it
        if self.random_split:
            softmax_scores_cut += random_noise

        #Sort the final softmax_scores
        #Take the negative
        final_softmax_scores_sorted, _ = torch.sort(-1 * softmax_scores_cut)

        #Get the threshold quantile
        #threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n
        threshold_quantile = np.ceil(((n + 1) * (self.alpha))) / n

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def _get_regularized_softmax_scores(self, softmax_scores_sorted, softmax_scores_sorted_indices, y_calibration=None):

        k = softmax_scores_sorted.size(1)
        n = softmax_scores_sorted.size(0)

        #If we have calibration data we use it

        # Check, maybe for calibration we do not need to use it
        if y_calibration is not None:
            # Use the true rank
            y_rank = torch.where(softmax_scores_sorted_indices==y_calibration.unsqueeze(1))[1].reshape(-1, 1)

        else:
            # Penalize everything above a certain rank
            y_rank = torch.ones(n, 1)*self.k_reg

        # Initialize the penalty vector
        regularization_penalty = torch.arange(k, dtype=float).expand(n, -1) - y_rank
        # For classes for which the rank is lower than the true class rank we have zero penalty
        regularization_penalty = torch.where(regularization_penalty < 0, torch.tensor(0), regularization_penalty)
        # Add the penalty (we need a negative score as we have smaller values, worse agreement
        regularization_penalty *= - self.lambda_penalty

        regularized_softmax_scores_sorted = softmax_scores_sorted + regularization_penalty
        return regularized_softmax_scores_sorted

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = -1 * F.softmax(logit_scores, dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, dim=1)

        if self.lambda_penalty > 0.:
            softmax_scores_sorted = self._get_regularized_softmax_scores(softmax_scores_sorted, softmax_scores_indices)

        prediction_sets = []

        for labels, values in zip(softmax_scores_indices, softmax_scores_sorted):
            prediction_set = []
            quantile = self.threshold_q.clone()

            cumulative_score = 0
            for label, value in zip(labels, values):
                if cumulative_score + value >= quantile:
                    prediction_set.append(label.item())
                    cumulative_score += value

                else:
                    #Avoid zero-set size
                    if len(prediction_set) == 0:
                        prediction_set.append(label.item())

                    break
            prediction_sets.append(prediction_set)

        return prediction_sets

def get_coverage(prediction_sets, dataset, test_set_mask, alpha, len_calibration_set):

    y_test = dataset.y[test_set_mask]
    n = len(y_test)
    coverage = 0
    #Check if label is contained in prediction_set
    for idx, set in enumerate(prediction_sets):
        if y_test[idx] in set:
            coverage += 1

    empirical_coverage = round(float(coverage/n), 4)

    # Check if we get approximately true coverage based on asymptotics
    variance = round((alpha * (1-alpha))/(len_calibration_set+2), 4)

    #assert empirical_coverage - round(3*np.sqrt(variance), 4) <= 1-alpha <= empirical_coverage + round(3*np.sqrt(variance), 4), "The coverage is not valid!"
    print(f"We have the coverage of {empirical_coverage} +- {round(np.sqrt(variance), 4)}")
    return empirical_coverage

def get_singleton_hit_ratio(prediction_sets, dataset, test_set_mask):

    y_test = dataset.y[test_set_mask]
    n = len(y_test)

    #Filter out prediction_sets larger than 1
    set_size_one = list(map(lambda x: len(x) == 1, prediction_sets))
    predicted_labels_singletons = list(filter(lambda x: len(x) == 1, prediction_sets))

    #Slice now list of lists y_test accordingly
    true_labels_singletons = [[true_label.item()] for true_label, singleton_flag in zip(y_test, set_size_one) if singleton_flag]

    correct_singletons = 0

    for predictions, true_labels in zip(predicted_labels_singletons, true_labels_singletons):
        if predictions == true_labels:
            correct_singletons += 1

    singleton_hit_ratio = round(float(correct_singletons/n), 4)

    return singleton_hit_ratio

def get_efficiency(prediction_sets):

    #len_prediction_sets = list(map(lambda x: len(x), prediction_sets))
    #Filter first zero/empty sets
    non_zero_prediction_sets = list(filter(lambda x: len(x) > 0, prediction_sets))
    len_non_zero_prediction_sets = list(map(lambda x: len(x), non_zero_prediction_sets))

    #Double-check: In principle, we should not have any zero-size sets
    #assert len(prediction_sets) == len(len_non_zero_prediction_sets)

    average_set_size = np.mean(len_non_zero_prediction_sets)

    return round(float(average_set_size), 4)

