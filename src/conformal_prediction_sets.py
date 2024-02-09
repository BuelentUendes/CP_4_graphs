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

#ToDo: Push to device!

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

        #Clip the trehsold quantile to avoid runtime errors
        threshold_quantile = np.clip(threshold_quantile, 0., 1.)

        #Sort the uncertainty scores from low to large
        uncertainty_scores_true_label_sorted = torch.sort(uncertainty_scores_true_label, descending=False)[0]

        return torch.quantile(uncertainty_scores_true_label_sorted, threshold_quantile, interpolation='higher')

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=-1)
        uncertainty_scores = 1 - softmax_scores

        #Get a mask to check which predictions are below the threshold
        uncertainty_scores_below_threshold = uncertainty_scores <= self.threshold_q

        self.prediction_sets = [torch.nonzero(mask).squeeze(dim=1).tolist() for mask in uncertainty_scores_below_threshold]

        return self.prediction_sets

class Adaptive_Conformer:

    def __init__(self, alpha, model, dataset, calibration_mask, random_split=False,
                 lambda_penalty=0., k_reg=None, constant_penalty=False, seed=7):

        seed_everything(seed)

        self.alpha = alpha
        self.model = model
        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.lambda_penalty = lambda_penalty
        self.k_reg = k_reg
        self.constant_penalty = constant_penalty
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

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)

        softmax_scores_cut = torch.tensor([cumulative_softmax_scores[sample, idx] for sample, idx in cutoff_idx]).reshape(-1, 1)

        # Get the random score
        if self.random_split:
            u_vec = torch.rand_like(softmax_score_true_class)
            ##Add the random noise to it (and subtract the softmax_true_class as we previously included it)
            # v * softmax -1 * softmax = (v-1) softmax_true_class
            softmax_scores_cut += (u_vec-1) * softmax_score_true_class
            random_noise = u_vec * softmax_score_true_class
            softmax_scores_cut += random_noise - softmax_score_true_class

        #Sort the final softmax_scores
        final_softmax_scores_sorted, _ = torch.sort(softmax_scores_cut, descending=True)

        #Get the threshold quantile
        threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n

        threshold_quantile = np.clip(threshold_quantile, 0.0, 1.0)

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def _get_regularized_softmax_scores(self, softmax_scores_sorted, softmax_scores_sorted_indices, y_calibration=None):

        k = softmax_scores_sorted.size(1)
        n = softmax_scores_sorted.size(0)

        #If we have calibration data we use it

        if y_calibration is not None:
            # Use the true rank
            y_rank = torch.where(softmax_scores_sorted_indices==y_calibration.unsqueeze(1))[1].reshape(-1, 1)

        else:
            # Penalize everything above a certain rank
            y_rank = torch.ones(n, 1)*self.k_reg

        # Initialize the penalty vector
        if self.constant_penalty:
            penalty = self.lambda_penalty

        # Linear penalty
        else:
            penalty = torch.arange(k, dtype=float) * self.lambda_penalty

        regularization_penalty = torch.arange(k, dtype=float).expand(n, -1) - y_rank

        # For classes for which the rank is lower than the true class rank we have zero penalty
        #regularization_penalty = torch.where(regularization_penalty < 0, torch.tensor(0), self.lambda_penalty)
        regularization_penalty = torch.where(regularization_penalty < 0, torch.tensor(0), penalty)

        regularized_softmax_scores_sorted = softmax_scores_sorted + regularization_penalty
        return regularized_softmax_scores_sorted

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, dim=1, descending=True)

        if self.lambda_penalty > 0.:
            softmax_scores_sorted = self._get_regularized_softmax_scores(softmax_scores_sorted, softmax_scores_indices)

        prediction_sets = []

        # #I can use torch.cumsum() and filter again here!
        cumulated_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        for cumulative_scores, softmax_scores_index in zip(cumulated_softmax_scores, softmax_scores_indices):
            try:
                cutoff_idx = torch.nonzero(cumulative_scores < self.threshold_q, as_tuple=False)[-1] #Get the last index just below the threshold
                prediction_set = softmax_scores_index[:min(len(softmax_scores_index), cutoff_idx + 1)].tolist() #Include plus 1

            except IndexError: #if all values are above the threshold, then I do have an empty set and I will return the first value above it
                prediction_set = [softmax_scores_index[0].item()]

            prediction_sets.append(prediction_set)

        return prediction_sets

class DAPS_threshold:
    """
    Implements DAPS on top of a threshold conformal prediction class as presented in the paper:
    https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf
    """
    # Code adapted from:
    # https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/cp/graph_transformations.py

    def __init__(self, alpha, model, dataset, calibration_mask, seed, neighborhood_coefficient):

        #super().__init__(alpha, model, dataset, calibration_mask, seed)

        seed_everything(seed)

        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.num_nodes = self.x.shape[0]
        self.neighborhood_coefficient = neighborhood_coefficient
        self.alpha = alpha
        self.model = model

        #Create the adjacency matrix and the degree matrix needed for the calculating the score
        self.A = torch.sparse.FloatTensor(
            dataset.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_nodes, self.num_nodes)
        )
        self.D = torch.matmul(self.A, torch.ones(self.A.shape[0]))

        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        #Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        #First step: Calculate the aggregated neighborhood scores
        neighborhood_logits = torch.linalg.matmul(self.A.to_dense(), logit_scores)
        neighborhood_logits_norm = neighborhood_logits / (1 / (self.D + 1e-10))[:, None]

        daps_scores = logit_scores * (1-self.neighborhood_coefficient) + self.neighborhood_coefficient * neighborhood_logits_norm

        softmax_scores = F.softmax(daps_scores[calibration_mask], dim=1)

        uncertainty_scores = 1 - softmax_scores
        true_label_indices = torch.arange(len(y_calibration))
        uncertainty_scores_true_label = uncertainty_scores[true_label_indices, y_calibration]

        threshold_quantile = np.ceil(((n+1)*(1-self.alpha)))/n

        #Clip the trehsold quantile to avoid runtime errors
        threshold_quantile = np.clip(threshold_quantile, 0., 1.)

        #Sort the uncertainty scores from low to large
        uncertainty_scores_true_label_sorted = torch.sort(uncertainty_scores_true_label, descending=False)[0]

        return torch.quantile(uncertainty_scores_true_label_sorted, threshold_quantile, interpolation='higher')

    def get_prediction_sets(self, test_set_mask):

        logit_scores = self.model(self.x, self.edge_index)[test_set_mask]
        softmax_scores = F.softmax(logit_scores, dim=-1)
        uncertainty_scores = 1 - softmax_scores

        #Get a mask to check which predictions are below the threshold
        uncertainty_scores_below_threshold = uncertainty_scores <= self.threshold_q

        self.prediction_sets = [torch.nonzero(mask).squeeze(dim=1).tolist() for mask in uncertainty_scores_below_threshold]

        return self.prediction_sets

class DAPS:
    """
    Implements the diffusion adaptive score on top of an APS as presented in the paper:
    https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf
    """
    def __init__(self, alpha, model, dataset, calibration_mask, seed, random_split, neighborhood_coefficient):

        #super().__init__(alpha, model, dataset, calibration_mask, seed)
        seed_everything(seed)

        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.num_nodes = self.x.shape[0]
        self.random_split = random_split
        self.neighborhood_coefficient = neighborhood_coefficient
        self.alpha = alpha
        self.model = model

        #Create the adjacency matrix and the degree matrix needed for the calculating the score
        self.A = torch.sparse.FloatTensor(
            dataset.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_nodes, self.num_nodes)
        )
        self.D = torch.matmul(self.A, torch.ones(self.A.shape[0]))

        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        #Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        #First step: Calculate the aggregated neighborhood scores
        neighborhood_logits = torch.linalg.matmul(self.A.to_dense(), logit_scores)
        neighborhood_logits_norm = neighborhood_logits * (1 / (self.D + 1e-10))[:, None]

        daps_scores = logit_scores * (1-self.neighborhood_coefficient) + self.neighborhood_coefficient * neighborhood_logits_norm

        softmax_scores = F.softmax(daps_scores[calibration_mask], dim=1)

        # Get the softmax score of the true class
        softmax_score_true_class = softmax_scores[torch.arange(softmax_scores.shape[0]), y_calibration].reshape(-1, 1)

        # Get the sorted probabilities from large to low
        softmax_scores_sorted, softmax_scores_sorted_indices = torch.sort(softmax_scores, descending=True, dim=1)

        # Now we summed up the probabilities
        cumulative_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)

        softmax_scores_cut = torch.tensor(
            [cumulative_softmax_scores[sample, idx] for sample, idx in cutoff_idx]).reshape(-1, 1)

        # Get the random score
        if self.random_split:
            u_vec = torch.rand_like(softmax_score_true_class)
            ## Add the random noise to it (and subtract the softmax_true_class as we previously included it)
            ## v * softmax -1 * softmax = (v-1) softmax_true_class
            softmax_scores_cut += (u_vec - 1) * softmax_score_true_class
            random_noise = u_vec * softmax_score_true_class
            softmax_scores_cut += random_noise - softmax_score_true_class

        # Sort the final softmax_scores
        final_softmax_scores_sorted, _ = torch.sort(softmax_scores_cut, descending=True)

        # Get the threshold quantile
        threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n

        threshold_quantile = np.clip(threshold_quantile, 0.0, 1.0)

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def get_prediction_sets(self, test_set_mask):

        #Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        #First step: Calculate the aggregated neighborhood scores
        neighborhood_logits = torch.linalg.matmul(self.A.to_dense(), logit_scores)
        neighborhood_logits_norm = neighborhood_logits * (1 / (self.D + 1e-10))[:, None]

        daps_scores = logit_scores * (1-self.neighborhood_coefficient) + self.neighborhood_coefficient * neighborhood_logits_norm

        softmax_scores = F.softmax(daps_scores[test_set_mask], dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, dim=1, descending=True)

        prediction_sets = []

        # #I can use torch.cumsum() and filter again here!
        cumulated_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        for cumulative_scores, softmax_scores_index in zip(cumulated_softmax_scores, softmax_scores_indices):
            try:
                cutoff_idx = torch.nonzero(cumulative_scores < self.threshold_q, as_tuple=False)[-1]
                prediction_set = softmax_scores_index[:min(len(softmax_scores_index), cutoff_idx + 1)].tolist()

            except IndexError:  # if all values are above the threshold, then I do have an empty set and I will return the first value above it
                prediction_set = [softmax_scores_index[0].item()]

            prediction_sets.append(prediction_set)

        return prediction_sets

class K_Hop_DAPS:
    """
    Implements the k-hop neighbors diffusion score as presented in the paper:
    https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf
    """
    def __init__(self, alpha, model, dataset, calibration_mask, seed,
                 random_split, k, neighborhood_coefficients):

        seed_everything(seed)

        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.num_nodes = self.x.shape[0]
        self.random_split = random_split

        assert len(neighborhood_coefficients) == k, "Please provide for each hop a coefficient in a list!"

        self.k = k
        self.neighborhood_coefficients = neighborhood_coefficients


        self.alpha = alpha
        self.model = model

        #Create the adjacency matrix and the degree matrix needed for the calculating the score
        self.A = torch.sparse.FloatTensor(
            dataset.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_nodes, self.num_nodes)
        )
        self.D = torch.matmul(self.A, torch.ones(self.A.shape[0]))

        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        #Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        #First we can get the
        k_hop_scores = self.neighborhood_coefficients[0] * logit_scores

        #ToDo: Check this one!
        for i in range(1, len(self.neighborhood_coefficients)):
            neighborhood_logits = torch.linalg.matmul(self.A.to_dense(), logit_scores)
            neighborhood_logits_norm = (neighborhood_logits * (1 / (self.D + 1e-10))[:, None])**i
            k_hop_scores += self.neighborhood_coefficients[i] * neighborhood_logits_norm

        softmax_scores = F.softmax(k_hop_scores[calibration_mask], dim=1)

        # Get the softmax score of the true class
        softmax_score_true_class = softmax_scores[torch.arange(softmax_scores.shape[0]), y_calibration].reshape(-1, 1)

        # Get the sorted probabilities from large to low
        softmax_scores_sorted, softmax_scores_sorted_indices = torch.sort(softmax_scores, descending=True, dim=1)

        # Now we summed up the probabilities
        cumulative_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)

        softmax_scores_cut = torch.tensor(
            [cumulative_softmax_scores[sample, idx] for sample, idx in cutoff_idx]).reshape(-1, 1)

        # Get the random score
        if self.random_split:
            u_vec = torch.rand_like(softmax_score_true_class)
            ##Add the random noise to it (and subtract the softmax_true_class as we previously included it)
            # v * softmax -1 * softmax = (v-1) softmax_true_class
            softmax_scores_cut += (u_vec - 1) * softmax_score_true_class
            random_noise = u_vec * softmax_score_true_class
            softmax_scores_cut += random_noise - softmax_score_true_class

        # Sort the final softmax_scores
        final_softmax_scores_sorted, _ = torch.sort(softmax_scores_cut, descending=True)

        # Get the threshold quantile
        threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n

        threshold_quantile = np.clip(threshold_quantile, 0.0, 1.0)

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def get_prediction_sets(self, test_set_mask):

        #Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        # First we can get the
        k_hop_scores = self.neighborhood_coefficients[0] * logit_scores

        for i in range(1, len(self.neighborhood_coefficients)):
            neighborhood_logits = torch.linalg.matmul(self.A.to_dense(), logit_scores)
            neighborhood_logits_norm = (neighborhood_logits * (1 / (self.D + 1e-10))[:, None]) ** i
            k_hop_scores += self.neighborhood_coefficients[i] * neighborhood_logits_norm

        softmax_scores = F.softmax(k_hop_scores[test_set_mask], dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, dim=1, descending=True)

        prediction_sets = []

        # #I can use torch.cumsum() and filter again here!
        cumulated_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        for cumulative_scores, softmax_scores_index in zip(cumulated_softmax_scores, softmax_scores_indices):
            try:
                cutoff_idx = torch.nonzero(cumulative_scores < self.threshold_q, as_tuple=False)[-1]
                prediction_set = softmax_scores_index[:min(len(softmax_scores_index), cutoff_idx + 1)].tolist()

            except IndexError:  # if all values are above the threshold, then I do have an empty set and I will return the first value above it
                prediction_set = [softmax_scores_index[0].item()]

            prediction_sets.append(prediction_set)

        return prediction_sets

class Score_Propagation:
    """
    Implements the score propagation (SP) algorithm as presented in the paper:
    https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf
    """

    def __init__(self, alpha, model, dataset, calibration_mask, seed,
                 random_split, neighborhood_coefficient, iterations=10):

        seed_everything(seed)

        self.x, self.edge_index, self.y = dataset.x, dataset.edge_index, dataset.y
        self.num_nodes = self.x.shape[0]
        self.random_split = random_split

        self.neighborhood_coefficient = neighborhood_coefficient
        self.iterations = iterations

        self.alpha = alpha
        self.model = model

        # Create the adjacency matrix and the degree matrix needed for the calculating the score
        self.A = torch.sparse.FloatTensor(
            dataset.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_nodes, self.num_nodes)
        )
        self.D = torch.matmul(self.A, torch.ones(self.A.shape[0]))

        self.threshold_q = self._get_quantile(calibration_mask)

    def _get_quantile(self, calibration_mask):

        y_calibration = self.y[calibration_mask]
        n = len(y_calibration)

        # Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        # Get first H0 score
        h0_score = (1-self.neighborhood_coefficient) * logit_scores
        # Store the score as we need this for the iteration
        h_score = h0_score.clone()

        for _ in range(self.iterations):
            degree_norm_adj = (self.A.to_dense()) * (1 / (self.D + 1e-10)[:, None])
            node_wise_score = torch.linalg.matmul(degree_norm_adj, h_score)
            #node_wise_score = torch.linalg.matmul(h_score, degree_norm_adj)
            h_score = h0_score + self.neighborhood_coefficient * node_wise_score

        softmax_scores = F.softmax(h_score[calibration_mask], dim=1)

        # Get the softmax score of the true class
        softmax_score_true_class = softmax_scores[torch.arange(softmax_scores.shape[0]), y_calibration].reshape(-1, 1)

        # Get the sorted probabilities from large to low
        softmax_scores_sorted, softmax_scores_sorted_indices = torch.sort(softmax_scores, descending=True, dim=1)

        # Now we summed up the probabilities
        cumulative_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        # Find the value where the target class occurs in the sorted index,
        # The first value is the sample and the second one the specific cutoff_idx
        cutoff_idx = torch.nonzero(softmax_scores_sorted_indices == y_calibration.unsqueeze(1), as_tuple=False)

        softmax_scores_cut = torch.tensor(
            [cumulative_softmax_scores[sample, idx] for sample, idx in cutoff_idx]).reshape(-1, 1)

        # Get the random score
        if self.random_split:
            u_vec = torch.rand_like(softmax_score_true_class)
            ##Add the random noise to it (and subtract the softmax_true_class as we previously included it)
            ## v * softmax -1 * softmax = (v-1) softmax_true_class
            softmax_scores_cut += (u_vec - 1) * softmax_score_true_class
            random_noise = u_vec * softmax_score_true_class
            softmax_scores_cut += random_noise - softmax_score_true_class

        # Sort the final softmax_scores
        final_softmax_scores_sorted, _ = torch.sort(softmax_scores_cut, descending=True)

        # Get the threshold quantile
        threshold_quantile = np.ceil(((n + 1) * (1 - self.alpha))) / n

        threshold_quantile = np.clip(threshold_quantile, 0.0, 1.0)

        return torch.quantile(final_softmax_scores_sorted, threshold_quantile)

    def get_prediction_sets(self, test_set_mask):

        # Get the logit scores of shape num_nodes, num_classes
        logit_scores = self.model(self.x, self.edge_index)

        # Get first H0 score
        h0_score = (1 - self.neighborhood_coefficient) * logit_scores
        # Store the score as we need this for the iteration
        h_score = h0_score.clone()

        for _ in range(self.iterations):
            degree_norm_adj = (self.A.to_dense()) / ((self.D + 1e-10)[:, None])
            node_wise_score = torch.linalg.matmul(degree_norm_adj, h_score)
            h_score = h0_score + self.neighborhood_coefficient * node_wise_score

        softmax_scores = F.softmax(h_score[test_set_mask], dim=1)

        softmax_scores_sorted, softmax_scores_indices = torch.sort(softmax_scores, dim=1, descending=True)

        prediction_sets = []

        # #I can use torch.cumsum() and filter again here!
        cumulated_softmax_scores = torch.cumsum(softmax_scores_sorted, dim=1)

        for cumulative_scores, softmax_scores_index in zip(cumulated_softmax_scores, softmax_scores_indices):
            try:
                cutoff_idx = torch.nonzero(cumulative_scores < self.threshold_q, as_tuple=False)[-1]
                prediction_set = softmax_scores_index[:min(len(softmax_scores_index), cutoff_idx + 1)].tolist()

            except IndexError:  # if all values are above the threshold, then I do have an empty set and I will return the first value above it
                prediction_set = [softmax_scores_index[0].item()]

            prediction_sets.append(prediction_set)

        return prediction_sets