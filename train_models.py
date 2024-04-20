########################################################################################
# Main script for training the models. Please read the README.MD for further information
# Author: Buelent Uendes
########################################################################################

# Import standard libraries
import warnings

warnings.filterwarnings("ignore")
import os

# Import the helper functions that are needed
from utils.helper_path import DATA_PATH, CONFIG_PATH, MODELS_PATH
from utils.helper import (
    get_data,
    DataManager,
    Graph_Trainer,
    load_yaml_config_file,
    get_model,
    seed_everything,
    create_directory,
)
from utils.helper_parse import (
    parse_dataset,
    parse_model,
    percentage_validator,
    homophily_validator,
)

# Import library for allowing terminal commands
import argparse

# Declare filenames needed for the script
MODELS_CONFIG_FILE_NAME = "models_config_file.yaml"
TRAINING_CONFIG_FILE_NAME = "training_config_file.yaml"
RESULTS_FILE_NAME = "results_model_training.csv"


def main(args):
    # Set the seed for reproducibility
    seed_everything(args.random_seed)

    model_config = load_yaml_config_file(
        os.path.join(CONFIG_PATH, MODELS_CONFIG_FILE_NAME)
    )
    training_config = load_yaml_config_file(
        os.path.join(CONFIG_PATH, TRAINING_CONFIG_FILE_NAME)
    )

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
            graph_trainer = Graph_Trainer(
                model, **training_config["training_protocols"][model_type]
            )
            if args.tuning_set:
                graph_trainer.fit(
                    dataset,
                    true_train_idx,
                    true_tuning_idx,
                    args.verbose,
                    **training_config["training_protocols"][model_type],
                )
            else:
                graph_trainer.fit(
                    dataset,
                    true_train_idx,
                    true_calibration_idx,
                    args.verbose,
                    **training_config["training_protocols"][model_type],
                )

            graph_trainer.test(dataset, test_idx)

            if args.save_model:
                save_name = f"{model_type}_{args.random_seed}.pt"
                # Create the folder to save the models
                save_location = os.path.join(MODELS_PATH, dataset_name)
                create_directory(save_location)
                graph_trainer.save_model(save_location, save_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--random_seed", help="Specify the seed", type=int, default=1
    )
    parser.add_argument(
        "--save_model",
        help="Boolean flag indicating if model should be saved. Default False",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="Boolean. Default True. If output should be printed",
        action="store_false",
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
