########################################################################################
# Main script for running the conformal prediction inference.
# Please make sure to run first the train_models.py file and save the respective models
# Please read the README.MD for further information
# Author: Buelent Uendes
########################################################################################

#Import standard libraries
import warnings
warnings.filterwarnings("ignore")
import os

#Import standard
import torch

# Import the helper functions that are needed
from utils.helper_path import DATA_PATH, CONFIG_PATH, MODELS_PATH
from utils.helper import get_data, DataManager, load_yaml_config_file, get_model, seed_everything, \
    get_singleton_hit_ratio, get_efficiency, get_coverage

from src.conformal_prediction_sets import Threshold_Conformer, Adaptive_Conformer, DAPS, K_Hop_DAPS, Score_Propagation

#Import library for allowing terminal commands
import argparse

def parse_dataset(arg):

    if arg == "all":
        return ["Cora", "Citeseer", "Pubmed", "Amazon-Computers", "Amazon-Photos",
                          "Coauthor-Physics", "Coauthor-CS", "Mixhop", "OGBN-Arxiv", "OGBN-Products"]
    else:
        datasets = arg.split(",")
        datasets = [dataset.strip() for dataset in datasets]
        return datasets

def parse_model(arg):

    if arg == "all":
        return ["GCN","GAT","SAGE","APPNPNet"]

    else:
        models = arg.split(",")
        models = [model.strip() for model in models]
        return models

def main(args):
    #Set the seed for reproducibility
    seed_everything(args.random_seed)

    model_config = load_yaml_config_file(os.path.join(CONFIG_PATH, args.models_config_file))
    #training_config = load_yaml_config_file(os.path.join(CONFIG_PATH, args.training_config_file))
    for dataset_name in args.dataset:
        #Retrieve the data
        dataset = get_data(dataset_name, root=DATA_PATH, homophily=args.homophily)

        num_features = dataset.x.shape[1]
        num_classes = len(dataset.y.unique())

        datamanager = DataManager(dataset, random_seed=args.random_seed)
        #Get train test split
        train_idx, test_idx = datamanager.get_train_test_split()
        true_train_idx, true_calibration_idx, true_tuning_idx = datamanager.get_calibration_split(train_idx,
                                                                                                  tuning_set=True,
                                                                                                  percentage_tuning_data=0.5)
        #Train each model
        for model_type in args.model:

            model = get_model(model_type, num_features, num_classes, **model_config["models_config"][model_type])
            save_name = f'{model_type}_{args.random_seed}.pt'
            # Get the location of the saved and trained model
            save_location = os.path.join(MODELS_PATH, dataset_name)
            try:
                model.load_state_dict(torch.load(os.path.join(save_location, save_name)))

            except FileNotFoundError:
                raise FileNotFoundError(f"Please run the 'train_models.py' with {model_type} for dataset {args.dataset} and seed {args.random_seed} first!")

            # Get the DAPS
            daps_conformer = DAPS(args.alpha, model, dataset, true_calibration_idx, random_split=args.random_split,
                                  seed=args.random_seed, neighborhood_coefficient=0.5)
            daps_prediction_sets = daps_conformer.get_prediction_sets(test_idx)

            # Get the 2-K neighborhood scores
            k_hop_conformer = K_Hop_DAPS(args.alpha, model, dataset, true_calibration_idx, random_split=args.random_split, k=2,
                                  seed=args.random_seed, neighborhood_coefficients=[0.8, 0.2])
            k_hop_prediction_sets = k_hop_conformer.get_prediction_sets(test_idx)

            # Get the score propagation scores
            score_propagation_conformer = Score_Propagation(args.alpha, model, dataset, true_calibration_idx, random_split=args.random_split,
                                  seed=args.random_seed, neighborhood_coefficient=0.85)
            score_propagation_prediction_sets = score_propagation_conformer.get_prediction_sets(test_idx)


            # Get the conformer now
            threshold_conformer = Threshold_Conformer(args.alpha, model, dataset, true_calibration_idx,
                                                      seed=args.random_seed)
            threshold_prediction_sets = threshold_conformer.get_prediction_sets(test_idx)
            adaptive_conformer = Adaptive_Conformer(args.alpha, model, dataset, true_calibration_idx,
                                                    random_split=args.random_split,
                                                    lambda_penalty=0., k_reg=2, seed=args.random_seed)
            adaptive_prediction_sets = adaptive_conformer.get_prediction_sets(test_idx)

            regularized_conformer = Adaptive_Conformer(args.alpha, model, dataset, true_calibration_idx,
                                                       random_split=args.random_split,
                                                       lambda_penalty=1, k_reg=1, constant_penalty=args.constant_penalty, seed=args.random_seed)
            regularized_prediction_sets = regularized_conformer.get_prediction_sets(test_idx)

            # Get the performance metrics
            empirical_coverage_threshold = get_coverage(threshold_prediction_sets, dataset, test_idx, args.alpha,
                                                        len(true_calibration_idx))
            empirical_coverage_adaptive = get_coverage(adaptive_prediction_sets, dataset, test_idx, args.alpha,
                                                       len(true_calibration_idx))
            empirical_coverage_regularized = get_coverage(regularized_prediction_sets, dataset, test_idx, args.alpha,
                                                          len(true_calibration_idx))
            empirical_coverage_daps = get_coverage(daps_prediction_sets, dataset, test_idx, args.alpha,
                                                   len(true_calibration_idx))
            empirical_coverage_k_hop = get_coverage(k_hop_prediction_sets, dataset, test_idx, args.alpha,
                                                   len(true_calibration_idx))
            empirical_coverage_score_propagation = get_coverage(score_propagation_prediction_sets, dataset, test_idx, args.alpha,
                                                    len(true_calibration_idx))

            singleton_hit_ratio_threshold = get_singleton_hit_ratio(threshold_prediction_sets, dataset, test_idx)
            efficiency_threshold = get_efficiency(threshold_prediction_sets)

            singleton_hit_ratio_adaptive = get_singleton_hit_ratio(adaptive_prediction_sets, dataset, test_idx)
            efficiency_adaptive = get_efficiency(adaptive_prediction_sets)

            singleton_hit_ratio_regularized = get_singleton_hit_ratio(regularized_prediction_sets, dataset, test_idx)
            efficiency_regularized = get_efficiency(regularized_prediction_sets)

            singleton_hit_ratio_daps = get_singleton_hit_ratio(daps_prediction_sets, dataset, test_idx)
            efficiency_daps = get_efficiency(daps_prediction_sets)

            singleton_hit_ratio_k_hop = get_singleton_hit_ratio(daps_prediction_sets, dataset, test_idx)
            efficiency_k_hop = get_efficiency(k_hop_prediction_sets)

            singleton_hit_ratio_score_propagation = get_singleton_hit_ratio(score_propagation_prediction_sets, dataset, test_idx)
            efficiency_score_propagation = get_efficiency(score_propagation_prediction_sets)

            print(f"The empirical coverage for threshold is {empirical_coverage_threshold}")
            print(f"The empirical coverage for adaptive is {empirical_coverage_adaptive}")
            print(f"The empirical coverage for regularized is {empirical_coverage_regularized}")
            print(f"The empirical coverage for DAPS is {empirical_coverage_daps}")
            print(f"The empirical coverage for K_Hop is {empirical_coverage_k_hop}")
            print(f"The empirical coverage for Score Propagation is {empirical_coverage_score_propagation}")
            print()

            print(f"The singleton hit ratio threshold is {singleton_hit_ratio_threshold}")
            print(f"The efficiency for threshold is {efficiency_threshold}")
            print()

            print(f"The singleton hit ratio adaptive is {singleton_hit_ratio_adaptive}")
            print(f"The efficiency for adaptive is {efficiency_adaptive}")
            print()

            print(f"The singleton hit ratio regularized is {singleton_hit_ratio_regularized}")
            print(f"The efficiency for regularized is {efficiency_regularized}")
            print()

            print(f"The singleton hit ratio DAPS is {singleton_hit_ratio_k_hop}")
            print(f"The efficiency for DAPS is {efficiency_daps}")
            print()

            print(f"The singleton hit ratio K_Hop is {singleton_hit_ratio_daps}")
            print(f"The efficiency for K_Hop is {efficiency_k_hop}")
            print()

            print(f"The singleton hit ratio Score Propagation is {singleton_hit_ratio_score_propagation}")
            print(f"The efficiency for Score Propagation is {efficiency_score_propagation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Specify which dataset to use", type=parse_dataset, default="Cora")
    parser.add_argument("--model", help="Specify which model(s) you want to train and use", type=parse_model, default="GCN")
    parser.add_argument("--random_seed", help="Specify the seed", type=int, default=1)
    parser.add_argument("--models_config_file", help="name of the model config file", default="models_config_file.yaml", type=str)
    parser.add_argument("--alpha", help="alpha value for the conformal prediction. 1-alpha is the coverage that one wants to achieve",
                        type=float, default=0.05)
    parser.add_argument("--constant_penalty", help="Applies constant penalty to the regularized version", action="store_true")
    parser.add_argument("--random_split", help="Boolean indicating if we want to randomly split. Default True", action="store_false")

    #We add the homophile argument. However, this is only used for the MixHop dataset
    parser.add_argument("--homophily", help="Level of homophily. This is only used for the 'Mixhop' dataset."
                                            "Needs to be in range [0.0 - 0.9]", default=0.6, type=float)

    args = parser.parse_args()

    #Start training
    main(args)




#ToDo: 31.01.2024
# Write RAPS -> Done! -> Double-check the logic!
# Write DAPS -> Done!

#ToDo 29.01.2024:
# Write split conformal prediction -> Done!
# Adaptive split conformal prediction -> Done!
# Siglehit and efficiency -> Done!
# Tune -> most results are not as good as the benchmarks as reported!


#ToDo: 26.01.2024
# Check in place to(self.device) is not inplace!
# Vectorize to make it a sparse matrix!

# Store results across architectures and seeds and average them and get the average and standard deviation performance
# Synthetic Cora -> Gradient-Gating for DEEP-MULTI RATE Learning on Graphs
# Write conformal prediction split procedure
# Write the diffusion score procedure
# Write performance metric scores; singlehit and efficiency
# Look into setup.py/toml installation script






