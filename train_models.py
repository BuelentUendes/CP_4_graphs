########################################################################################
# Main script for training the models. Please read the README.MD for further information
# Author: Buelent Uendes
########################################################################################

#Import standard libraries
import warnings
warnings.filterwarnings("ignore")
import os

# Import the helper functions that are needed
from utils.helper_path import DATA_PATH, CONFIG_PATH, MODELS_PATH
from utils.helper import get_data, DataManager, Graph_Trainer, load_yaml_config_file, get_model, \
    seed_everything, create_directory

#Import library for allowing terminal commands
import argparse

def parse_dataset(arg):

    if arg == "all":
        return ["Cora", "Citeseer", "Pubmed", "Amazon-Computers", "Amazon-Photo",
                          "Coauthor-Physics", "Coauthor-CS", "Mixhop", "OGBN-Arxiv", "OGBN-Products"]
    else:
        datasets = arg.split(",")
        datasets = [dataset.strip() for dataset in datasets]
        return datasets

def parse_model(arg):

    if arg == "all":
        return ["GCN", "GAT", "SAGE", "APPNPNet"]

    else:
        models = arg.split(",")
        models = [model.strip() for model in models]
        return models

def main(args):
    #Set the seed for reproducibility
    seed_everything(args.random_seed)

    model_config = load_yaml_config_file(os.path.join(CONFIG_PATH, args.models_config_file))
    training_config = load_yaml_config_file(os.path.join(CONFIG_PATH, args.training_config_file))

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
        
        true_train_idx, true_calibration_idx = datamanager.get_calibration_split(train_idx, tuning_set=False, percentage_tuning_data=0.5)
        
        #Train each model
        for model_type in args.model:

            model = get_model(model_type, num_features, num_classes, **model_config["models_config"][model_type])
            graph_trainer = Graph_Trainer(model, **training_config["training_protocols"][model_type])
            graph_trainer.fit(dataset, true_train_idx, true_tuning_idx, args.verbose, **training_config["training_protocols"][model_type])
            graph_trainer.test(dataset, test_idx)

            if args.save_model:
                save_name = f'{model_type}_{args.random_seed}.pt'
                #Create the folder to save the models
                save_location = os.path.join(MODELS_PATH, dataset_name)
                create_directory(save_location)
                graph_trainer.save_model(save_location, save_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Specify which dataset to use", type=parse_dataset, default="Amazon-Photo")
    parser.add_argument("--model", help="Specify which model(s) you want to train and use", type=parse_model, default="GCN")
    parser.add_argument("--random_seed", help="Specify the seed", type=int, default=1)
    parser.add_argument("--models_config_file", help="name of the model config file", default="models_config_file.yaml", type=str)
    parser.add_argument("--training_config_file", help="name of the training config_file", default="training_config_file.yaml", type=str)
    parser.add_argument("--save_model", help="Boolean flag indicating if model should be saved", action="store_true")
    parser.add_argument("--verbose", help="Boolean. Default False. If output should be printed", action="store_false")

    #We add the homophile argument. However, this is only used for the MixHop dataset
    parser.add_argument("--homophily", help="Level of homophily. This is only used for the 'Mixhop' dataset."
                                            "Needs to be in range [0.0 - 0.9]", default=0.6, type=float)

    args = parser.parse_args()

    #Start training
    main(args)

