# Source code for conformal prediction

import numpy as np
import torch
import torch.nn.functional as F

class Threshold_Conformer:

    def __init__(self, alpha, model, dataset, calibration_mask):

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

        #uncertainty_scores_sorted, _ = torch.sort(uncertainty_scores_true_label)
        # With uncertainty score
        threshold_quantile = np.ceil(((n+1)*(1-self.alpha)))/n

        #Calculate the variance to make sure  we are still in the bounds
        self.variance = (self.alpha * (1-self.alpha))/(n+2)

        return torch.quantile(uncertainty_scores_true_label, threshold_quantile)

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=-1)

        #Get a mask for all predictions that manage to get over the threshold
        softmax_above_threshold = softmax_scores >= 1 - self.threshold_q

        self.prediction_sets = [torch.nonzero(mask).squeeze(dim=1).tolist() for mask in softmax_above_threshold]

        return self.prediction_sets

class Adaptive_Conformer:

    def __init__(self, alpha, model, dataset, calibration_mask):

        self.alpha = alpha
        self.model = model
        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        logit_scores = self.model(self.x, self.edge_index)[calibration_mask]
        softmax_scores = F.softmax(logit_scores, dim=1)

        # Get the sorted probabilities from low to large
        softmax_scores_sorted, softmax_scores_sorted_indices = torch.sort(softmax_scores, descending=True, dim=1)

        # Now we summed up the probabilities
        cumulative_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)
        # Now sum up the probabilities until the true class is detected

        softmax_scores_cut = torch.tensor([cumulative_softmax_scores[sample, idx] for sample,idx in cutoff_idx])

        #Sort the final softmax_scores
        final_softmax_scores_sorted, _ = torch.sort(softmax_scores_cut)

        #Get the threshold quantile
        threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, descending=True, dim=1)

        prediction_sets = []

        for labels, values in zip(softmax_scores_indices, softmax_scores_sorted):
            prediction_set = []
            score = self.threshold_q.clone()

            for label, value in zip(labels, values):
                if score - value > 0:
                    prediction_set.append(label.item())
                    score -= value

                else:
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
    variance = (alpha * (1-alpha))/(len_calibration_set+2)

    assert empirical_coverage - 3*np.sqrt(variance) <= 1-alpha <= empirical_coverage + 3*np.sqrt(variance), "The coverage is not valid!"
    #print(f"We have the coverage of {empirical_coverage} +- {np.sqrt(variance)}")
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

    len_prediction_sets = list(map(lambda x: len(x), prediction_sets))

    average_set_size = np.mean(len_prediction_sets)

    return round(float(average_set_size), 4)

