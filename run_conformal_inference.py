########################################################################################
# Main script for running the conformal prediction inference.
# Please make sure to run first the train_models.py file and save the respective models
# Please read the README.MD for further information
# Author: Buelent Uendes
########################################################################################

# Import standard libraries
import warnings

warnings.filterwarnings("ignore")
# Import library for allowing terminal commands
import argparse
import os
import csv

# Import standard
import torch

from src.conformal_prediction_sets import (
    DAPS,
    Adaptive_Conformer,
    K_Hop_DAPS,
    Score_Propagation,
    Threshold_Conformer,
)
from utils.helper import (
    create_directory,
    DataManager,
    get_coverage,
    get_data,
    get_efficiency,
    get_model,
    get_singleton_hit_ratio,
    load_yaml_config_file,
    seed_everything,
)
from utils.helper_parse import (
    parse_dataset,
    parse_model,
    percentage_validator,
    homophily_validator,
)

# Import the helper functions that are needed
from utils.helper_path import CONFIG_PATH, DATA_PATH, MODELS_PATH, RESULTS_PATH

# Declare filenames needed for the script
CONFORMAL_PREDICTION_CONFIG_FILE_NAME = "conformal_prediction_config.yaml"
RESULTS_FILE_NAME = "results_conformal_prediction.csv"


def main(args):
    # Set the seed for reproducibility
    seed_everything(args.random_seed)
    model_config = load_yaml_config_file(
        os.path.join(CONFIG_PATH, args.models_config_file)
    )
    conformal_config = load_yaml_config_file(
        os.path.join(CONFIG_PATH, CONFORMAL_PREDICTION_CONFIG_FILE_NAME)
    )
    # Create results directory in case it is not created yet
    create_directory(RESULTS_PATH)

    with open(os.path.join(RESULTS_PATH, RESULTS_FILE_NAME), mode="a", newline="") as results_file:
        fieldnames = [
            "Dataset",
            "Model",
            "Empirical Coverage (Threshold)",
            "Empirical Coverage (Adaptive)",
            "Empirical Coverage (Regularized)",
            "Empirical Coverage (DAPS)",
            "Empirical Coverage (K_Hop)",
            "Empirical Coverage (Score Propagation)",
            "Singleton Hit Ratio (Threshold)",
            "Efficiency (Threshold)",
            "Singleton Hit Ratio (Adaptive)",
            "Efficiency (Adaptive)",
            "Singleton Hit Ratio (Regularized)",
            "Efficiency (Regularized)",
            "Singleton Hit Ratio (DAPS)",
            "Efficiency (DAPS)",
            "Singleton Hit Ratio (K_Hop)",
            "Efficiency (K_Hop)",
            "Singleton Hit Ratio (Score Propagation)",
            "Efficiency (Score Propagation)"
        ]
        results_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        results_writer.writeheader()

        for dataset_name in args.dataset:
            # Retrieve the data
            dataset = get_data(
                dataset_name, root=DATA_PATH, homophily=args.homophily
            )

            num_features = dataset.x.shape[1]
            num_classes = len(dataset.y.unique())

            datamanager = DataManager(dataset, random_seed=args.random_seed)
            # Get train test split
            train_idx, test_idx = datamanager.get_train_test_split()
            if args.tuning_set:
                true_train_idx, true_calibration_idx, true_tuning_idx = (
                    datamanager.get_calibration_split(
                        train_idx, tuning_set=args.tuning_set, percentage_tuning_data=args.percentage_tuning_data
                    )
                )
            else:
                true_train_idx, true_calibration_idx = (
                    datamanager.get_calibration_split(
                        train_idx, tuning_set=args.tuning_set, percentage_tuning_data=args.percentage_tuning_data
                    )
                )
            # Train each model
            for model_type in args.model:

                model = get_model(
                    model_type,
                    num_features,
                    num_classes,
                    **model_config["models_config"][model_type],
                )
                save_name = f"{model_type}_{args.random_seed}.pt"
                # Get the location of the saved and trained model
                save_location = os.path.join(MODELS_PATH, dataset_name)
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(save_location, save_name))
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Please run the 'train_models.py' with {model_type} for dataset {args.dataset} "
                        f"and seed {args.random_seed} first!"
                    )

                # Get the DAPS
                daps_conformer = DAPS(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    random_tie_breaking=conformal_config["DAPS"]["random_tie_breaking"],
                    seed=args.random_seed,
                    neighborhood_coefficient=conformal_config["DAPS"]["neighborhood_coefficient"],
                )
                daps_prediction_sets = daps_conformer.get_prediction_sets(test_idx)

                # Get the 2-K neighborhood scores
                k_hop_conformer = K_Hop_DAPS(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    random_tie_breaking=conformal_config["K_Hop_DAPS"]["random_tie_breaking"],
                    k=conformal_config["K_Hop_DAPS"]["k"],
                    seed=args.random_seed,
                    neighborhood_coefficients=conformal_config["K_Hop_DAPS"]["neighborhood_coefficients"],
                )
                k_hop_prediction_sets = k_hop_conformer.get_prediction_sets(
                    test_idx
                )

                # Get the score propagation scores
                score_propagation_conformer = Score_Propagation(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    random_tie_breaking=conformal_config["Score_Propagation"]["random_tie_breaking"],
                    seed=args.random_seed,
                    neighborhood_coefficient=conformal_config["Score_Propagation"]["neighborhood_coefficient"],
                )
                score_propagation_prediction_sets = (
                    score_propagation_conformer.get_prediction_sets(test_idx)
                )

                # Get the conformer now
                threshold_conformer = Threshold_Conformer(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    seed=args.random_seed,
                )
                threshold_prediction_sets = (
                    threshold_conformer.get_prediction_sets(test_idx)
                )
                adaptive_conformer = Adaptive_Conformer(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    random_tie_breaking=conformal_config["Adaptive_Conformer"]["random_tie_breaking"],
                    lambda_penalty=conformal_config["Adaptive_Conformer"]["lambda_penalty"],
                    k_reg=conformal_config["Adaptive_Conformer"]["k_reg"],
                    constant_penalty=conformal_config["Adaptive_Conformer"]["constant_penalty"],
                    seed=args.random_seed,
                )
                adaptive_prediction_sets = adaptive_conformer.get_prediction_sets(
                    test_idx
                )

                regularized_conformer = Adaptive_Conformer(
                    args.alpha,
                    model,
                    dataset,
                    true_calibration_idx,
                    random_tie_breaking=conformal_config["Regularized_Conformer"]["random_tie_breaking"],
                    lambda_penalty=conformal_config["Regularized_Conformer"]["lambda_penalty"],
                    k_reg=conformal_config["Regularized_Conformer"]["k_reg"],
                    constant_penalty=conformal_config["Regularized_Conformer"]["constant_penalty"],
                    seed=args.random_seed,
                )
                regularized_prediction_sets = (
                    regularized_conformer.get_prediction_sets(test_idx)
                )

                # Get the performance metrics
                empirical_coverage_threshold = get_coverage(
                    threshold_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )
                empirical_coverage_adaptive = get_coverage(
                    adaptive_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )
                empirical_coverage_regularized = get_coverage(
                    regularized_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )
                empirical_coverage_daps = get_coverage(
                    daps_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )
                empirical_coverage_k_hop = get_coverage(
                    k_hop_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )
                empirical_coverage_score_propagation = get_coverage(
                    score_propagation_prediction_sets,
                    dataset,
                    test_idx,
                    args.alpha,
                    len(true_calibration_idx),
                )

                singleton_hit_ratio_threshold = get_singleton_hit_ratio(
                    threshold_prediction_sets, dataset, test_idx
                )
                efficiency_threshold = get_efficiency(threshold_prediction_sets)

                singleton_hit_ratio_adaptive = get_singleton_hit_ratio(
                    adaptive_prediction_sets, dataset, test_idx
                )
                efficiency_adaptive = get_efficiency(adaptive_prediction_sets)

                singleton_hit_ratio_regularized = get_singleton_hit_ratio(
                    regularized_prediction_sets, dataset, test_idx
                )
                efficiency_regularized = get_efficiency(
                    regularized_prediction_sets
                )

                singleton_hit_ratio_daps = get_singleton_hit_ratio(
                    daps_prediction_sets, dataset, test_idx
                )
                efficiency_daps = get_efficiency(daps_prediction_sets)

                singleton_hit_ratio_k_hop = get_singleton_hit_ratio(
                    daps_prediction_sets, dataset, test_idx
                )
                efficiency_k_hop = get_efficiency(k_hop_prediction_sets)

                singleton_hit_ratio_score_propagation = get_singleton_hit_ratio(
                    score_propagation_prediction_sets, dataset, test_idx
                )
                efficiency_score_propagation = get_efficiency(
                    score_propagation_prediction_sets
                )

                results_writer.writerow({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Empirical Coverage (Threshold)": empirical_coverage_threshold,
                    "Empirical Coverage (Adaptive)": empirical_coverage_adaptive,
                    "Empirical Coverage (Regularized)": empirical_coverage_regularized,
                    "Empirical Coverage (DAPS)": empirical_coverage_daps,
                    "Empirical Coverage (K_Hop)": empirical_coverage_k_hop,
                    "Empirical Coverage (Score Propagation)": empirical_coverage_score_propagation,
                    "Singleton Hit Ratio (Threshold)": singleton_hit_ratio_threshold,
                    "Efficiency (Threshold)": efficiency_threshold,
                    "Singleton Hit Ratio (Adaptive)": singleton_hit_ratio_adaptive,
                    "Efficiency (Adaptive)": efficiency_adaptive,
                    "Singleton Hit Ratio (Regularized)": singleton_hit_ratio_regularized,
                    "Efficiency (Regularized)": efficiency_regularized,
                    "Singleton Hit Ratio (DAPS)": singleton_hit_ratio_daps,
                    "Efficiency (DAPS)": efficiency_daps,
                    "Singleton Hit Ratio (K_Hop)": singleton_hit_ratio_k_hop,
                    "Efficiency (K_Hop)": efficiency_k_hop,
                    "Singleton Hit Ratio (Score Propagation)": singleton_hit_ratio_score_propagation,
                    "Efficiency (Score Propagation)": efficiency_score_propagation
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--random_seed", help="Specify the seed", type=int, default=1
    )
    parser.add_argument(
        "--dataset",
        help="Specify which dataset to use",
        type=parse_dataset,
        default="Cora",
    )
    parser.add_argument(
        "--model",
        help="Specify which model(s) you want to train and use",
        type=parse_model,
        default="GCN",
    )
    parser.add_argument(
        "--models_config_file",
        help="name of the model config file",
        default="models_config_file.yaml",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        help="alpha value for the conformal prediction. 1-alpha is the coverage that one wants to achieve",
        default=0.10,
        type=percentage_validator,
    )
    parser.add_argument(
        "--tuning_set",
        help="Boolean. If set, then we allow for a tuning set. Default: False",
        action="store_true",
    )
    parser.add_argument(
        "--percentage_tuning_data",
        help="Percentage of tuning data as a percentage of training data",
        default=0.5,
        type=percentage_validator,
    )
    parser.add_argument(
        "--homophily",
        help="Level of homophily. This is only used for the 'Mixhop' dataset."
             "Needs to be in range [0.0 - 0.9]",
        default=0.6,
        type=homophily_validator,
    )

    args = parser.parse_args()

    # Start training
    main(args)
